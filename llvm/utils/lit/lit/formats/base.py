import abc

class TestFormat(object):
    """Base class for test formats.

    A TestFormat encapsulates logic for finding and executing a certain type of
    test. For example, a subclass FooTestFormat would contain the logic for
    finding tests written in the 'Foo' format, and the logic for running a
    single one.

    TestFormat is an Abstract Base Class (ABC). It uses the Python abc.ABCMeta
    type and associated @abc.abstractmethod decorator. Together, these provide
    subclass behaviour which is notionally similar to C++ pure virtual classes:
    only subclasses which implement all abstract methods can be instantiated
    (the implementation may come from an intermediate base).

    For details on ABCs, see: https://docs.python.org/2/library/abc.html. Note
    that Python ABCs have extensive abilities beyond what is used here. For
    TestFormat, we only care about enforcing that abstract methods are
    implemented.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def getTestsInDirectory(self, testSuite, path_in_suite, litConfig,
                            localConfig):
      """Finds tests of this format in the given directory.

      Args:
          testSuite: a Test.TestSuite object.
          path_in_suite: the subpath under testSuite to look for tests.
          litConfig: the LitConfig for the test suite.
          localConfig: a LitConfig with local specializations.

      Returns:
          An iterable of Test.Test objects.
      """

    @abc.abstractmethod
    def execute(self, test, litConfig):
      """Runs the given 'test', which is of this format.

      Args:
          test: a Test.Test object describing the test to run.
          litConfig: the LitConfig for the test suite.

      Returns:
          A tuple of (status:Test.ResultCode, message:str)
      """
