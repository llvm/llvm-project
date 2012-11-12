import os

# Test results.

class TestResult:
    def __init__(self, name, isFailure):
        self.name = name
        self.isFailure = isFailure

PASS        = TestResult('PASS', False)
XFAIL       = TestResult('XFAIL', False)
FAIL        = TestResult('FAIL', True)
XPASS       = TestResult('XPASS', True)
UNRESOLVED  = TestResult('UNRESOLVED', True)
UNSUPPORTED = TestResult('UNSUPPORTED', False)

# Test classes.

class TestFormat:
    """TestFormat - Test information provider."""

    def __init__(self, name):
        self.name = name

class TestSuite:
    """TestSuite - Information on a group of tests.

    A test suite groups together a set of logically related tests.
    """

    def __init__(self, name, source_root, exec_root, config):
        self.name = name
        self.source_root = source_root
        self.exec_root = exec_root
        # The test suite configuration.
        self.config = config

    def getSourcePath(self, components):
        return os.path.join(self.source_root, *components)

    def getExecPath(self, components):
        return os.path.join(self.exec_root, *components)

class Test:
    """Test - Information on a single test instance."""

    def __init__(self, suite, path_in_suite, config):
        self.suite = suite
        self.path_in_suite = path_in_suite
        self.config = config
        # The test result code, once complete.
        self.result = None
        # Any additional output from the test, once complete.
        self.output = None
        # The wall time to execute this test, if timing and once complete.
        self.elapsed = None
        # The repeat index of this test, or None.
        self.index = None

    def copyWithIndex(self, index):
        import copy
        res = copy.copy(self)
        res.index = index
        return res

    def setResult(self, result, output, elapsed):
        assert self.result is None, "Test result already set!"
        self.result = result
        self.output = output
        self.elapsed = elapsed

    def getFullName(self):
        return self.suite.config.name + ' :: ' + '/'.join(self.path_in_suite)

    def getSourcePath(self):
        return self.suite.getSourcePath(self.path_in_suite)

    def getExecPath(self):
        return self.suite.getExecPath(self.path_in_suite)
