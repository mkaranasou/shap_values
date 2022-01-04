if __name__ == '__main__':
    feature_names = [f'f{i}' for i in range(6)]
    spark = get_spark_session()
    df = generate_df_for_features(feature_names, 5000)
    assembler = VectorAssembler(
        inputCols=feature_names,
        outputCol="features")
    df = assembler.transform(df)
    df.show(10, False)
    df = add_labels(df)
    (trainingData, testData) = df.randomSplit([0.8, 0.2])

    estimator = DecisionTreeClassifier(
        labelCol="label", featuresCol="features",
    )
    # estimator = RandomForestClassifier(
    #     labelCol="label", featuresCol="features", numTrees=10
    # )
    model = estimator.fit(trainingData)
    predictions = model.transform(testData)
    print(predictions.display())
    column_to_examine = 'prediction'
    predictions.select(column_to_examine, "label", "features").show(5)

    evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol=column_to_examine
    )
    accuracy = evaluator.evaluate(predictions)
    print(f'Test Error = %{(1.0 - accuracy)}')

    testData = select_row(testData, testData.select('id').take(1)[0].id)
    row_of_interest = testData.select('id', 'features').where(
        F.col('is_selected') == True  # noqa
    ).first()
    print('Row: ', row_of_interest)
    testData = testData.select('*').where(F.col('is_selected') != True)
    df = df.drop(
        column_to_examine
    )
    