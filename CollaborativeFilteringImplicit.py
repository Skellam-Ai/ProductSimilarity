# Databricks notebook source
""" 
Collaborative Filtering to get the Propensity Score between Items

MODELS USED:
1. NEIGHBORHOOD BASED ON USERS
2. LATENT FACTOR MODEL - ALTERNATING LEAST SQAURES


OUTPUT : ITEM - ITEM SIMILARITY SCORE, MODEL TYPE


"""



# COMMAND ----------

# Import libraries  - KNN - CF
from pyspark.sql.functions import count, avg,sum
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.linalg import Vectors
from pyspark.mllib.linalg import Vectors
from pyspark.sql import Row
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.sql.types import *
import timeit
from pyspark.sql.functions import lit
import logging

# Import libraries for Matric Factorization
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.sql import SQLContext
from pyspark.sql.types import IntegerType,DoubleType
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

from pyspark.sql.types import DoubleType
from pyspark.ml.feature import IndexToString
from pyspark.mllib.linalg.distributed import *

from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.mllib.linalg import Vectors as MLLibVectors
from pyspark.sql.functions import lit





logging.basicConfig( format='%(name)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
#logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger("py4j").setLevel(logging.ERROR)
job_start_time = timeit.timeit()
logging.info('Job Started...')
logging.info('loading Libraries...')

# COMMAND ----------

# PARAMETERS

params = {
  "output_path_knn":"/mnt/workspaces/CustomerAnalytics/dev/DeepBrew/CollaborativeFiltering/UserKNN/July8/Test",
  "output_path_als":"/mnt/workspaces/CustomerAnalytics/dev/DeepBrew/CollaborativeFiltering/ALS/July8/Test"
}


# COMMAND ----------

# DATA READ
# usuals Qty path for UAT XIDS
#usualsQtyJsonFileXid = "/mnt/workspaces/CustomerAnalytics/dev/DeepBrew/usuals/UsualsJsonFullDataAllXidsQtyJune7v4"
#usualsQtyRawFileXid = "/mnt/workspaces/CustomerAnalytics/dev/DeepBrew/usuals/Usuals1yrAllXIDSRawDeltaQtyJune7"


# usuals Qty path for Entire Data All XIDS for 1 yr
#usualsQtyJsonFile = "/mnt/workspaces/CustomerAnalytics/dev/DeepBrew/usuals/UsualsJsonFullDataAllXidsQtyJune12"
#usualsQtyRawFile = "/mnt/workspaces/CustomerAnalytics/dev/DeepBrew/usuals/Usuals1yrAllXIDSRawDeltaQtyJune12"


# Usuals Raw Transactions for the specified user
# read the dataframe from the file
users_raw_df = spark.read.parquet(usualsQtyRawFileXid)
#users_raw_df = spark.read.parquet(usualsQtyRawFile)
#display(users_raw_df)

# COMMAND ----------

##  DATA PREPROCESSING
# Filter out modifiers - they ight not be needed
users_raw_df = users_raw_df.filter("ProductType != 'ProductOption'")

# Get the unique products list
prod_details_df = users_raw_df.select("SKU","ProductNumber","FormCode","SizeCode","ProductName","ItemId").dropDuplicates()

# Transform POS Data to User - Item - Timestamp - Frequency
users_item_tm_df = users_raw_df.groupby('XID','SKU','TransTmstmpDtm').agg(sum('NetRevQty').alias('qty'))

# Transform POS Data to User - Item - Frequency
user_item_df = users_item_tm_df.groupby('XID','SKU').agg(sum('qty').alias('freq'))


# DATA PRE-PROCESSING FOR NEIGHBORHOOD METHOD
# Transform POS Data to User - Item - Frequency Matrix
user_prod_df = user_item_df.groupby('XID').pivot('SKU').agg(sum('freq').alias('freq'))
user_prod_df = user_prod_df.limit(200)


# Drop XID
user_prod_df_noxid = user_prod_df.drop('XID')

# Product list
item_tuple_list = user_prod_df_noxid.dtypes
item_name_str_list =  [item_name for item_name,_ in item_tuple_list]
item_name_list =  [int(item_name) for item_name,_ in item_tuple_list]

# Fill the NA with 0
user_item_data = user_prod_df_noxid.fillna(0)

# Drop Duplicate rows
user_item_data  = user_item_data.dropDuplicates()


# COMMAND ----------


##  NEIGHBORHOOD BASED COLLABORATIVE FILTERING

def neighborhood_cf(user_item_data,prod_details_df,item_name_str_list,item_name_list):
  # Convert Product Columns into Features
  assembler = VectorAssembler(
    inputCols=item_name_str_list,
    outputCol="features")

  user_item_vec = assembler.transform(user_item_data).select("features")
  user_item_vec  = user_item_vec.dropDuplicates()

  # Scale the Features
  scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")

  # Compute summary statistics and generate MinMaxScalerModel
  scalerModel = scaler.fit(user_item_vec)

  # rescale each feature to range [min, max].
  scaledDataDf = scalerModel.transform(user_item_vec).select("scaledFeatures")
  print("Count of Users : {}".format(scaledDataDf.count()))
  scaledDataDf  = scaledDataDf.dropDuplicates()
  print("Count of Users : {}".format(scaledDataDf.count()))

  # Convert scaled dataframe to RDD of Vector
  data_rdd = scaledDataDf.rdd.map(lambda row: MLLibVectors.dense(row["scaledFeatures"]))

  # Transform into row matrix
  # Create a RowMatrix from an RDD of vectors.
  mat = RowMatrix(data_rdd)
  # Get its size.
  m = mat.numRows()  # 4
  n = mat.numCols() 
  print("Size of the User - Item Matrix")
  print(m,n)

  # Calculate exact Simalarities for a given Item - find the similar Item
  exact_sim = mat.columnSimilarities()
  exact_sim_df = exact_sim.entries.toDF()

  # Get the Product SKU for the similairty matrix
  cSchema = StructType([StructField("index", IntegerType()),StructField("SKU", IntegerType())])
  item_name_list2 = [[i,item_name] for i,item_name in enumerate(item_name_list)]
  # dataframe containing item-index
  item_df = spark.createDataFrame(item_name_list2,schema = cSchema)
  exact_sim_df = exact_sim_df.join(item_df, exact_sim_df.i == item_df.index,'inner').withColumnRenamed("SKU","SKU1").drop("index")
  item_sim_df = exact_sim_df.join(item_df, exact_sim_df.j == item_df.index,'inner').withColumnRenamed("SKU","SKU2").drop("index").drop("i").drop("j")
  item_sim_df = item_sim_df.join(prod_details_df,item_sim_df.SKU1 == prod_details_df.SKU,'inner').withColumnRenamed("ProductName","ProductName1").select("SKU1","SKU2","ProductName1","value")
  item_sim_df = item_sim_df.join(prod_details_df,item_sim_df.SKU2 == prod_details_df.SKU,'inner').withColumnRenamed("ProductName","ProductName2").select("SKU1","SKU2","ProductName1","ProductName2","value")
  item_sim_df = item_sim_df.withColumn("ModelType",lit("USER-KNN"))
  
  return item_sim_df



# COMMAND ----------



def get_nearest_item(item_sku,item_sim_df,prod_details_df):
  sku_filter = item_sku
  filter_item = "SKU1 == {}".format(sku_filter)
  item_sim_df_filtered = item_sim_df.filter(filter_item)
  nearest_item = item_sim_df_filtered.orderBy('value', ascending=False).limit(1).select("SKU2").collect()[0].SKU2
  prod_res_df = prod_details_df.where(prod_details_df.SKU.isin([nearest_item]))
  nearest_item_name = prod_res_df.select("ProductName").collect()[0].ProductName
  print("Nearest Item : {}".format(nearest_item_name))
  
  return nearest_item_name

def get_nearest_items(item_sku,item_sim_df,prod_details_df):
  sku_filter = item_sku
  filter_item = "SKU1 == {}".format(sku_filter)
  item_sim_df_filtered = item_sim_df.filter(filter_item)
  nearest_items = item_sim_df_filtered.orderBy('value', ascending=False)
  nearest_items_df = nearest_items.join(prod_res_df,nearest_items.SKU2 == prod_details_df.SKU,'inner')
  
  return nearest_items_df


def get_farthest_item(item_sku,item_sim_df,prod_details_df):
  sku_filter = item_sku
  filter_item = "SKU1 == {}".format(sku_filter)
  item_sim_df_filtered = item_sim_df.filter(filter_item)
  farthest_item = item_sim_df_filtered.orderBy('value', ascending=True).limit(1).select("SKU2").collect()[0].SKU2
  prod_res_df = prod_details_df.where(prod_details_df.SKU.isin([farthest_item]))
  farthest_item_name = prod_res_df.select("ProductName").collect()[0].ProductName
  print("Farthest Item : {}".format(farthest_item_name))
  
  return farthest_item_name



  

# COMMAND ----------

# Example usage of getting nearest / farthest item
#item_sku = 110567
#nearest_item_name = get_nearest_item(item_sku,item_sim_df,prod_details_df)
#farthest_item_name = get_farthest_item(item_sku,item_sim_df,prod_details_df)


# COMMAND ----------

# MATRIX FACTORIZATION BASED COLLABORATIVE FILTERING

# Data Preprocessing for MATRIX FACTORIZATION

def als_cf(users_item_tm_df,prod_details_df):
  df = users_item_tm_df
  df = df.limit(200) # limit - optional
  df.cache()
  # Convert DF to RDD
  ratingsRDD = df.rdd.map(lambda p: Row(userId=str(p[0]), prodId=int(p[1]),
                                     freq=float(p[3]), timestamp=str(p[2])))
  ratings = spark.createDataFrame(ratingsRDD)

  # Index the User ID into appropriate numbers - Integer Format
  stringIndexer = StringIndexer(inputCol="userId", outputCol="userId1",handleInvalid="error")
  model_str_ind = stringIndexer.fit(ratings)
  ratings_ind = model_str_ind.transform(ratings)
  ratings_freq = ratings_ind.withColumn('userId2',ratings_ind.userId1.cast(IntegerType())).withColumn('prodId1',ratings_ind.prodId.cast(IntegerType()))
  ratings_freq = ratings_freq.select("userId2","prodId1","freq")
  #ratings3 = ratings3.limit(200) # limit - optional
  ratings_freq.cache()
  #display(ratings3)

  # split the data into Test and Train - Optional
  (training, test) = ratings_freq.randomSplit([0.8, 0.2])

  # COLLABORATIVE FILTERING USING ALTERNATING LEAST SQUARES
  # Build the recommendation model using ALS on the training data
  # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
  als = ALS(rank=10,maxIter=50, regParam=0.01, implicitPrefs=True,alpha=30,userCol="userId2", itemCol="prodId1", ratingCol="freq")
  model = als.fit(training)

  # Evaluate the model by computing the RMSE on the test data
  predictions = model.transform(test).fillna(0)
  #display(predictions)
  predictions1 = predictions.withColumn('userId2',predictions.userId2.cast(DoubleType()))
  # evaluation  metrics
  evaluator = RegressionEvaluator(metricName="rmse", labelCol="freq",
                                predictionCol="prediction")
  rmse = evaluator.evaluate(predictions)
  #logger.info("Root-mean-square error = " + str(rmse))

  # get the Item feature vectors from the model
  ifact = model.itemFactors
  array_to_vec = udf(lambda vs: Vectors.dense(vs), VectorUDT())
  item_vec_cf = ifact.withColumn("featureVec",array_to_vec("features")).drop("features")

  # Scale the Features
  scaler2 = MinMaxScaler(inputCol="featureVec", outputCol="scaledFeatures")

  # Compute summary statistics and generate MinMaxScalerModel
  scalerModel2 = scaler2.fit(item_vec_cf)

  # rescale each feature to range [min, max].
  scaledDataDf2 = scalerModel2.transform(item_vec_cf).select("id","scaledFeatures")

  user_item_sim_df = IndexedRowMatrix(scaledDataDf2.rdd.map(lambda x:(x[0],MLLibVectors.dense(x[1].toArray())))).toBlockMatrix(2,2).transpose().toIndexedRowMatrix().columnSimilarities().entries.toDF()

  user_item_sim_res = user_item_sim_df.withColumn("ModelType",lit("ALS")).withColumnRenamed("i","SKU1").withColumnRenamed("j","SKU2")
  user_item_sim_res = user_item_sim_res.join(prod_details_df,user_item_sim_res.SKU1 == prod_details_df.SKU,'inner').withColumnRenamed("ProductName","ProductName1").select("SKU1","SKU2","ProductName1","value","ModelType")
  user_item_sim_res = user_item_sim_res.join(prod_details_df,user_item_sim_res.SKU2 == prod_details_df.SKU,'inner').withColumnRenamed("ProductName","ProductName2").select("SKU1","SKU2","ProductName1","ProductName2","value","ModelType")
  
  return user_item_sim_res


# COMMAND ----------

## USER - USER NEIGHBORHOOD BASED ITEM SIMILARITY
# get the item - item similarity from User Neighborhood based Methods
item_sim_df_knn = neighborhood_cf(user_item_data,prod_details_df,item_name_str_list,item_name_list)

# write the similarity file 
try:
  print("User-KNN")
  #item_sim_df.write(params["output_path_knn"])
except Exception as e:
  logging.error("DATA WRITE : Problem in writing knn result Data !", exc_info=True)
  raise
  


# COMMAND ----------



## ALTERNATING LEAST SQUARE BASED ITEM SIMILARITY
# get the item -item similarity from ALTERNATING LEAST SQUARE  based Methods
item_sim_df_als = als_cf(users_item_tm_df,prod_details_df)

# write the similarity file 
try:
  print("User-KNN")
  #item_sim_df_als.write(params["output_path_als"])
except Exception as e:
  logging.error("DATA WRITE : Problem in writing als result Data !", exc_info=True)
  raise
  

# COMMAND ----------

display(item_sim_df_knn)

# COMMAND ----------

display(item_sim_df_als)

# COMMAND ----------


res_knn = spark.read.parquet(params["output_path_knn"])
res_als = spark.read.parquet(params["output_path_als"])

# COMMAND ----------

display(res_knn)

# COMMAND ----------

display(res_als)
