# test familiarity module using clusters
# Code Level 4: directory is implemented as third arg, allowing database
# to be in any specified directory.

import numpy as np
import tensorflow as tf
import sqlite3
from pathlib import Path
from random import randint
from random import seed
from datetime import datetime
from datetime import timedelta
from uuid import uuid4
from knet import initNeuralNet
from knet import stringToArray

def test_tweets(user, datex, directory, cur, fd=None):
    time1 = datetime.now()
# ------------------------------ section 1 ------------------------------------

    nInputs = 50
    nHidden = 20
    nOutputs = nInputs
    numTest = 10 # only numTest out of max_tweet_row in single_cluster_date_vectors will be run

    sql = "select cluster_id from cluster where user_id = ?"
    cur.execute(sql, (user,))
    (cluster,) = cur.fetchone()

    # If table single_cluster_date_vectors exists from a previous run, delete it

    cur.execute("drop table if exists single_cluster_date_vectors")

    time2 = datetime.now()
    print('\nsec 1: initialization time =', str(time2-time1), '\n')
    fd.write(totalSeconds(time2-time1) + ', ')

# ------------------------------ section 2 ------------------------------------

    sql = """
           create table single_cluster_date_vectors as
           select t.tweet_id, t.vector
           from tweet t, cluster c
           where t.created_at_date = ? and t.user_id = c.user_id and c.cluster_id = ?
           """
    cur.execute(sql, (datex,cluster))

    time3 = datetime.now()
    print('\nsec 2: create table single_cluster_date_vectors time =', str(time3-time2), '\n')
    fd.write(totalSeconds(time3-time2) + ', ')

# ------------------------------ section 3 ------------------------------------

    sql = "select count(*) from single_cluster_date_vectors"
    cur.execute(sql)
    (max_tweet_row,) = cur.fetchone()
    print("max_tweet_row in single_cluster_date_vectors = ", max_tweet_row)

    if max_tweet_row == 0:  # user u has no tweets on datex
        return None

    model = initNeuralNet(nInputs, nHidden, nOutputs)

    x_test = np.zeros( [numTest, nInputs] )
    y_test = np.zeros( [numTest, nOutputs] )

    for e in range(numTest):  # 0 to numTest-1
        tweet_row = randint(1, max_tweet_row) # row of single_custer_vectors for example e

        sql = """
              select tweet_id, vector from single_cluster_date_vectors
              where rowid = ?
              """
        cur.execute(sql, (tweet_row,)) # tweet_row is random # between 1 and max_tweet_row
        (tweet_id, vector) = cur.fetchone()

        if vector is None:
            continue

        print("test tweet_id =", tweet_id)

        # load this tweet into example e in testing set
        vect = stringToArray(vector)

        for i in range(len(vect)):
            x_test[e, i] = vect[i]      # input
            y_test[e, i] = x_test[e, i] # target = input

    time4 = datetime.now()
    fd.write(totalSeconds(time4-time3) + ', ')

# ------------------------------ section 4 ------------------------------------

    # load weights from previous training of the model
    model.load_weights('weightsForCluster_' + cluster + '.wt')

    # test the model
    print("\ntesting of model:\n")

    results = model.evaluate(x_test, y_test)

    for i in range(len(model.metrics_names)):
        print(model.metrics_names[i], " : ", results[i])

    sql = "drop table single_cluster_date_vectors"
    cur.execute(sql);

    time5 = datetime.now()
    print('\nsec 4: load weights & run neural net test time =', str(time5-time4), '\n')
    print('\ntotal time for function test_tweets =', str(time5-time1), '\n')

    fd.write(totalSeconds(time5-time4) + ', ')
    fd.write(totalSeconds(time5-time1) + '\n')

    return results[1]

# -----------------------------------------------------------------------------


def totalSeconds(td): # timedelta td in seconds as a string
    tsec = td / timedelta(seconds=1)
    return str(tsec)

def timings(numfc): # number of calls on test_tweets to time
    fd = open("test_keras_timings.csv", "w")
    fd.write("section1, section2, section3, section4, totaltime\n")

    directory = "/home/ronmintz/TwitterCodeLevels/CodeLevel4/data_directory"
    con = sqlite3.connect(directory + "/sim.sqlite3")
    cur = con.cursor()

    sql = "select count(*) from user_date"
    cur.execute(sql)
    (nRows,) = cur.fetchone()

    seed(20)

    for fc in range(numfc):  # 0 to numfc-1
        row = randint(1, nRows) # row of user_date

        sql = """
              select user_id, created_at_date from user_date
              where rowid = ?
              """
        cur.execute(sql, (row,)) # row is random # between 1 and nRows
        (user, datex) = cur.fetchone()

        test_tweets(user, datex, directory, cur, fd)

    fd.close()

