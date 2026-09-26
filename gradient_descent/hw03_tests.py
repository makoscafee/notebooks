import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def rv(value_list):
    """
    Takes a list of numbers and returns a row vector: 1 x n
    """
    return np.array([value_list])


def cv(value_list):
    """
    Takes a list of numbers and returns a column vector:  n x 1
    """
    return np.transpose(rv(value_list))


def lin_reg(X, th, th0):
    """
    Gets predictions from linear regression

    x (np.array): n-by-d
    th (np.array): d-by-m where m is the number of different settings of th
    th0 (np.array): 1-by-m
    Returns n-by-m np.array of predictions
    """
    return np.dot(X, th) + th0


def mse(x, y, th, th0):
    return np.mean((y - lin_reg(x, th, th0)) ** 2, axis=0, keepdims=True)


def ridge_obj(x, y, th, th0, lam):
    return mse(x, y, th, th0) + lam * np.linalg.norm(th) ** 2


def ridge_analytic(X, y, lam):
    """
    Determine best theta using the analytic (closed-form) expression for ridge regression.

    X (np.array): n-by-d array of features
    y (np.array): n-by-1 array of targets
    lam (float): lambda i.e. L2-regularizer coefficient

    Returns: d-by-1 array of theta values
    """
    n, d = X.shape
    return np.linalg.inv(X.T @ X + n * lam * np.identity(d)) @ X.T @ y


def cross_validate_analytic(X, y, lam):
    n, d = X.shape
    total_loss = 0
    kf = KFold(n_splits=5)

    for train_index, test_index in kf.split(X, y=y):
        X_train_split, y_train_split = X[train_index], y[train_index]
        X_val_split, y_val_split = X[test_index], y[test_index]
        X_train_split, y_train_split, X_val_split, y_val_split = (
            standardize_train_and_eval(
                X_train_split, y_train_split, X_val_split, y_val_split
            )
        )
        theta = ridge_analytic(X_train_split, y_train_split, lam)
        val_loss = np.mean((X_val_split @ theta - y_val_split) ** 2)
        total_loss += val_loss
    return total_loss / kf.n_splits


def standardize_train_and_eval(X_train, y_train, X_eval, y_eval):
    """Fit scalers on training data and transform both data splits."""
    
    # Standardize each feature to zero mean and unit variance.
    scaler_x = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)
    return (
        scaler_x.transform(X_train),
        scaler_y.transform(y_train),
        scaler_x.transform(X_eval),
        scaler_y.transform(y_eval),
    )


def get_data_splits_with_transforms(data, selected_features, output_feature):
    # data is an n x d matrix
    # selected_features is a list of strings identifying the features we want to select
    # output_feature is the target variable (string)

    X = data[selected_features].values
    y = data[output_feature].values

    # Split before fitting transforms so test data does not affect training.
    train_proportion = 0.8
    test_index = int(train_proportion * X.shape[0])
    X_train = np.copy(X[0:test_index, :])
    X_test = np.copy(X[test_index:, :])
    y_train = np.copy(y[0:test_index, :])
    y_test = np.copy(y[test_index:, :])

    return standardize_train_and_eval(X_train, y_train, X_test, y_test)


### Testing code


def TestCase(inputs, expected, fn=None):
    """
    Runs a single test case.
    """
    result = inputs[0]
    if fn:
        result = fn(*inputs)

    shape_error = False
    try:
        if np.allclose(result, expected, atol=1e-3):
            if isinstance(expected, np.ndarray):
                assert np.array_equal(expected.shape, result.shape)
            print("Test case passed!")
            print("----------------------------")
            return True
    except AssertionError:
        shape_error = True
    except:
        pass

    # display single inputs in a nicer way
    if len(inputs) == 1:
        inputs = inputs[0]

    if not fn:
        # don't show function inputs if not testing a function
        print("FAILED")
    else:
        print("FAILED on inputs:\n", inputs)

    print("Expected:\n", expected)
    print("Got:\n", result)

    if shape_error:
        print("Hint: Check the shapes of your output.")
    print("----------------------------")
    return False


def RunTestSuite(cases, fn=None, seed=None):
    """
    Cases is a list of lists (or other 2D iterable) where:
        * each row is a list/tuple/iterable, representing a test case
        * last element of each row is the expected answer
        * all elements before are the inputs to the function to check, in order
        * if the last element (expected answer) is a np array object, the
          student's answer will also be checked for getting shape correct.

    fn is the function to check. if fn=None, only the first element of Cases
       will be checked to see if it is np.allclose() to the last element.
    """
    passed = 0
    failed = 0
    for entries in cases:
        if seed is not None:
            np.random.seed(seed)
        expected = entries[-1]
        input = entries[:-1]
        if TestCase(input, expected, fn):
            passed += 1
        else:
            failed += 1
    if passed == len(cases):
        print("\nAll tests passed!")
    else:
        print("\nRan %d tests: %d passed, %d failed." % (len(cases), passed, failed))


def f1(x):
    return x[1:2, :] ** 2 + x[2:3, :]


def df1(x):
    x = list(x.squeeze())
    return cv([0, 2 * x[1], 1])


def f2(x):
    return x[0:1, :] * x[1:2, :]


def df2(x):
    x = list(x.squeeze())
    return cv([x[1], x[0], 0])


def package_ans(gd_vals):
    x, f = gd_vals
    return [x.tolist(), float(np.asarray(f).squeeze())]


def TestGradientDescent(fn):
    expected1 = [
        [[1.0], [1.2302319221611202e-97], [-98.99999999999865]],
        [[-98.99999999999865]],
    ]
    actual1 = fn(f1, df1, cv([1.0, 1.0, 1.0]), lambda x: 0.1, 1000)
    print("Test 1:")
    if np.allclose(expected1[0], actual1[0].tolist()) and np.isclose(
        float(np.asarray(actual1[1]).squeeze()),
        float(np.asarray(expected1[1]).squeeze()),
        atol=1.0e-4,
    ):
        print("Passed!")
    else:
        print("Test 1 failed with f(x) = x[1:2, :]**2 + x[2:3, :]")
    expected2 = [
        [[-10479.577710978716], [10479.577926834949], [4.0]],
        [[-109821551.26252407]],
    ]
    actual2 = fn(f2, df2, cv([2.0, 3.0, 4.0]), lambda x: 0.01, 1000)
    print("Test 2:")
    if np.allclose(actual2[0].tolist(), expected2[0]) and np.isclose(
        float(np.asarray(actual2[1]).squeeze()),
        float(np.asarray(expected2[1]).squeeze()),
        atol=1.0e-4,
    ):
        print("Passed!")
    else:
        print("Test 2 failed with f(x) = x[0:1, :]*x[1:2, :]")


def TestNumGrad(fn):
    cases = [
        (
            f1,
            cv([1.0, 1.0, 1.0]),
            np.array(([[0.0], [2.0000000000000018], [0.9999999999998899]])),
        ),
        (
            f1,
            cv([1.0, 2.0, 3.0]),
            np.array(([[0.0], [3.9999999999995595], [0.9999999999994458]])),
        ),
        (
            f2,
            cv([-1.0, -1.0, -1.0]),
            np.array(([[-0.9999999999999454], [-0.9999999999999454], [0.0]])),
        ),
        (
            f2,
            cv([-1.0, -2.0, -3.0]),
            np.array(([[-1.9999999999998908], [-0.9999999999998899], [0.0]])),
        ),
    ]
    RunTestSuite(cases, lambda f, x: fn(f)(x))


def TestMinimize(fn):
    print("Test 1")
    ans = package_ans(fn(f1, cv([1.0, 1.0, 1.0]), lambda i: 0.1, 1000))
    expected = [
        [[1.0], [2.3564483697668948e-14], [-99.00000000010797]],
        [[-99.00000000010797]],
    ]
    if np.allclose(ans[0], expected[0]) and np.allclose(ans[1], expected[1]):
        print("Passed")
    else:
        print("Test 1 failed with f(x) = x[1:2, :]**2 + x[2:3, :]")

    print("Test 2")
    ans = package_ans(fn(f2, cv([2.0, 3.0, 4.0]), lambda i: 0.01, 1000))
    expected = [
        [[-10479.577713756074], [10479.57792981472], [4.0]],
        [[-109821551.32285637]],
    ]
    if np.allclose(ans[0], expected[0]) and np.allclose(ans[1], expected[1]):
        print("Passed")
    else:
        print("Test 2 failed with f(x) = x[0:1, :]*x[1:2, :]")


def TestSGD(sgd_func, num_grad):
    """
    Test for stochastic gradient descent sgd()

    Requires two arguments: your `sgd` function and your `make_num_grad_fn` function
    """

    def J(Xi, yi, w):
        # translate from (1-augmented X, y, theta) to (separated X, y, th, th0) format
        return ridge_obj(Xi[:, :-1], yi, w[:-1, :], w[-1:, :].T, 0).T

    def dJ(Xi, yi, w):
        def f(w):
            return J(Xi, yi, w)

        return num_grad(f)(w)

    X = np.array(
        [
            [0.0, 0.1, 0.2, 0.3, 0.42, 0.52, 0.72, 0.78, 0.84, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    ).T
    y = np.array([[0.4, 0.6, 1.2, 0.1, 0.22, -0.6, -1.5, -0.5, -0.5, 0.0]]).T

    cases = [
        (
            X,
            y,
            J,
            dJ,
            cv([0.0, 0.0]),
            lambda i: 0.1,
            1000,
            np.array([[-1.438266235911751], [0.7366695997841343]]),
        ),
        (
            X,
            y,
            J,
            dJ,
            cv([-0.1, 0.1]),
            lambda i: 0.01,
            1000,
            np.array([[-1.1713461515169397], [0.5626086484724836]]),
        ),
    ]
    RunTestSuite(cases, sgd_func, seed=0)


def TestObjectiveFunc(objective_func):
    X_1 = np.array([[5, 5, 10, 16, 0]]).T
    Y_1 = np.array([[16, 14, 5, 1, 10]]).T

    X_2 = np.array([[6, 18, 12, 19, 15], [4, 7, 3, 10, 12], [16, 8, 15, 4, 6]]).T
    Y_2 = np.array([[0, 6, 15, 10, 16]]).T

    theta_1 = np.array([[3]])
    theta_2 = np.array([[5], [7], [3]])
    J_1_noreg = objective_func(X_1, Y_1, 0)(theta_1)
    assert isinstance(J_1_noreg, float), "Wrong return type"

    cases = [
        (X_1, Y_1, 0, theta_1, 587.2),
        (X_1, Y_1, 1, theta_1, 596.2),
        (X_2, Y_2, 2, theta_2, 20569.2),
    ]
    RunTestSuite(cases, lambda x, y, lam, th: objective_func(x, y, lam)(th))


def TestObjectiveFuncGrad(objective_func_grad):
    X_1 = np.array([[5, 5, 10, 16, 0]]).T
    Y_1 = np.array([[16, 14, 5, 1, 10]]).T

    X_2 = np.array([[6, 18, 12, 19, 15], [4, 7, 3, 10, 12], [16, 8, 15, 4, 6]]).T
    Y_2 = np.array([[0, 6, 15, 10, 16]]).T

    theta_1 = np.array([[3]])
    theta_2 = np.array([[5], [7], [3]])

    cases = [
        (X_1, Y_1, 0, theta_1, np.array([[400.8]])),
        (X_1, Y_1, 1, theta_1, np.array([[406.8]])),
        (X_2, Y_2, 2, theta_2, np.array([[4172.8], [2211.2000000000003], [2512.4]])),
    ]
    RunTestSuite(cases, lambda x, y, lam, th: objective_func_grad(x, y, lam)(th))


def TestRidgeGD(fn):
    X_train = np.array(
        [
            [2.367446386606739, 8.7520044953452, 8.14485148030185, 6.516389972287224],
            [
                0.039733614908928905,
                3.68480648450503,
                5.42213838170197,
                6.672864650109663,
            ],
            [
                3.9301662224536074,
                0.6049574778006783,
                1.7312268364620098,
                0.8299188668622759,
            ],
        ]
    ).T
    y_train = np.array(
        [[0.31776829626931735, 7.180994658970237, 2.402211421436035, 1.300292687028518]]
    ).T
    theta_actual = [[1.2447193743504172], [-1.0131608964646763], [-0.7209261159877435]]
    cases = [(X_train, y_train, 0.01, lambda i: 0.01, theta_actual)]
    RunTestSuite(cases, fn)
