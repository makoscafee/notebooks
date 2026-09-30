import numpy as np


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


def separable_medium():
    X = np.array([[2, -1, 1, 1], [-2, 2, 2, -1]]).T
    Y = np.array([[1, 0, 1, 0]]).T
    return X, Y


def super_simple_separable():
    X = np.array([[2, 3, 9, 12], [5, 2, 6, 5]]).T
    Y = np.array([[1, 0, 1, 0]]).T
    return X, Y


def test_llc_obj(sigmoid, nll_loss, llc_obj):
    """
    Note that these three tests test each of the three functions in order.
    If you want to test earlier functions but not later ones, you can pass None for those args.
    """
    sep_e_separator = np.array([[-0.40338351], [1.1849563]]), np.array([[-2.26910091]])
    X_1, Y_1 = super_simple_separable()
    th1, th1_0 = sep_e_separator
    ans = sigmoid(0.5).tolist()
    #    print(ans)
    expected = 0.6224593312018546
    if np.isclose(ans, expected):
        print("Test 1 passed")
    else:
        print("Test 1 for sigmoid failed")

    ans = nll_loss(X_1, Y_1, th1, th1_0).tolist()
    #    print(ans)
    expected = [
        [0.0562926158181702], [0.2850112020940751], [0.26093403339997306], [0.266793037070205]
    ]
    if np.allclose(ans, expected):
        print("Test 2 passed")
    else:
        print("Test 2 for nll_loss failed")

    ans = llc_obj(X_1, Y_1, th1, th1_0, 0.1)
    #    print(ans)
    expected = 0.37394169100056684
    if np.isclose(ans, expected) and isinstance(ans, float):
        print("Test 3 passed")
    else:
        print("Test 3 for llc_obj failed")


def test_llc_grad(
    d_sigmoid, d_nll_loss_th, d_nll_loss_th0, d_llc_obj_th, d_llc_obj_th0, llc_obj_grad
):
    """
    Note that these tests test each of the inputs functions in order.
    If you want to test earlier functions but not later ones, you can pass None for those args.
    """
    X2, y2 = super_simple_separable()
    th2, th20 = np.array([[-3.0, 15.0]]).T, np.array([[2.0]])

    ans = d_sigmoid(np.array([[71, -23.0]])).tolist()
    #    print(ans)
    expected = [[0.0, 1.0261879629595766e-10]]
    if np.allclose(ans, expected):
        print("Test 1 passed")
    else:
        print("Test 1 for d_sigmoid failed")

    ans = d_nll_loss_th(X2, y2, th2, th20).tolist()
    #    print(ans)
    expected = [
        [0.0, 2.9999999996921436, 0.0, 12.0],
        [0.0, 1.9999999997947624, 0.0, 5.0],
    ]
    if np.allclose(ans, expected):
        print("Test 2 passed")
    else:
        print("Test 2 for d_nll_loss_th failed")

    ans = d_nll_loss_th0(X2, y2, th2, th20).tolist()
    #    print(ans)
    expected = [[0.0, 0.9999999998973812, 0.0, 1.0]]
    if np.allclose(ans, expected):
        print("Test 3 passed")
    else:
        print("Test 3 for d_nll_loss_th0 failed")

    ans = d_llc_obj_th(X2, y2, th2, th20, 0.01).tolist()
    #    print(ans)
    expected = [[3.6899999999230357], [2.0499999999486906]]
    if np.allclose(ans, expected):
        print("Test 4 passed")
    else:
        print("Test 4 for d_llc_obj_th failed")

    ans = d_llc_obj_th0(X2, y2, th2, th20, 0.01).tolist()
    #    print(ans)
    expected = [[0.4999999999743453]]
    if np.allclose(ans, expected):
        print("Test 5 passed")
    else:
        print("Test 5 for d_llc_obj_th0 failed")

    ans = llc_obj_grad(X2, y2, th2, th20, 0.01).tolist()
    #    print(ans)
    expected = [[3.6899999999230357], [2.0499999999486906], [0.4999999999743453]]
    if np.allclose(ans, expected):
        print("Test 6 passed")
    else:
        print("Test 6 for llc_obj_grad failed")


def test_llc_min(llc_min):
    x_1, y_1 = super_simple_separable()
    ans = llc_min(x_1, y_1, 0.0001)
    #    print(ans)
    expected = [
        [[-1.897383932746237], [3.3962308285238088], [-0.3185808665118801]],
        0.30406118305471325,
    ]
    if np.allclose(ans[0], expected[0]) and np.isclose(ans[1], expected[1]):
        print("Test 1 passed")
    else:
        print("Test 1 on simple dataset failed")

    x_1, y_1 = separable_medium()
    ans = llc_min(x_1, y_1, 0.0001)
    #    print(ans)
    expected = [
        [[1.377362757219987], [0.3661947799990386], [-0.7768724769832256]],
        0.380271389072302,
    ]
    if np.allclose(ans[0], expected[0]) and np.isclose(ans[1], expected[1]):
        print("Test 2 passed")
    else:
        print("Test 2 on medium dataset failed")


def TestBinaryLoss(fn):
    cases = [
        (
            np.array([[[1, 0, 1]], [[1, 1, 0]]]),
            np.array([[[1, 1, 1]], [[1, 0, 0]]]),
            np.array([[1 / 3], [1 / 3]]),
        ),
        (
            np.array([[[1, 0, 1, 0, 1, 1]], [[1, 1, 0, 1, 0, 0]]]),
            np.array([[[1, 0, 1, 1, 0, 0]], [[1, 0, 0, 0, 0, 1]]]),
            np.array([[0.5], [0.5]]),
        ),
    ]
    RunTestSuite(cases, fn)
