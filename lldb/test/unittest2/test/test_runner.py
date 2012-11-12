import pickle

from cStringIO import StringIO
from unittest2.test.support import LoggingResult, OldTestResult

import unittest2


class Test_TextTestRunner(unittest2.TestCase):
    """Tests for TextTestRunner."""

    def test_init(self):
        runner = unittest2.TextTestRunner()
        self.assertFalse(runner.failfast)
        self.assertFalse(runner.buffer)
        self.assertEqual(runner.verbosity, 1)
        self.assertTrue(runner.descriptions)
        self.assertEqual(runner.resultclass, unittest2.TextTestResult)


    def testBufferAndFailfast(self):
        class Test(unittest2.TestCase):
            def testFoo(self):
                pass
        result = unittest2.TestResult()
        runner = unittest2.TextTestRunner(stream=StringIO(), failfast=True,
                                           buffer=True)
        # Use our result object
        runner._makeResult = lambda: result
        runner.run(Test('testFoo'))
        
        self.assertTrue(result.failfast)
        self.assertTrue(result.buffer)

    def testRunnerRegistersResult(self):
        class Test(unittest2.TestCase):
            def testFoo(self):
                pass
        originalRegisterResult = unittest2.runner.registerResult
        def cleanup():
            unittest2.runner.registerResult = originalRegisterResult
        self.addCleanup(cleanup)
        
        result = unittest2.TestResult()
        runner = unittest2.TextTestRunner(stream=StringIO())
        # Use our result object
        runner._makeResult = lambda: result
        
        self.wasRegistered = 0
        def fakeRegisterResult(thisResult):
            self.wasRegistered += 1
            self.assertEqual(thisResult, result)
        unittest2.runner.registerResult = fakeRegisterResult
        
        runner.run(unittest2.TestSuite())
        self.assertEqual(self.wasRegistered, 1)
        
    def test_works_with_result_without_startTestRun_stopTestRun(self):
        class OldTextResult(OldTestResult):
            def __init__(self, *_):
                super(OldTextResult, self).__init__()
            separator2 = ''
            def printErrors(self):
                pass

        runner = unittest2.TextTestRunner(stream=StringIO(), 
                                          resultclass=OldTextResult)
        runner.run(unittest2.TestSuite())

    def test_startTestRun_stopTestRun_called(self):
        class LoggingTextResult(LoggingResult):
            separator2 = ''
            def printErrors(self):
                pass

        class LoggingRunner(unittest2.TextTestRunner):
            def __init__(self, events):
                super(LoggingRunner, self).__init__(StringIO())
                self._events = events

            def _makeResult(self):
                return LoggingTextResult(self._events)

        events = []
        runner = LoggingRunner(events)
        runner.run(unittest2.TestSuite())
        expected = ['startTestRun', 'stopTestRun']
        self.assertEqual(events, expected)

    def test_pickle_unpickle(self):
        # Issue #7197: a TextTestRunner should be (un)pickleable. This is
        # required by test_multiprocessing under Windows (in verbose mode).
        import StringIO
        # cStringIO objects are not pickleable, but StringIO objects are.
        stream = StringIO.StringIO("foo")
        runner = unittest2.TextTestRunner(stream)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            s = pickle.dumps(runner, protocol=protocol)
            obj = pickle.loads(s)
            # StringIO objects never compare equal, a cheap test instead.
            self.assertEqual(obj.stream.getvalue(), stream.getvalue())

    def test_resultclass(self):
        def MockResultClass(*args):
            return args
        STREAM = object()
        DESCRIPTIONS = object()
        VERBOSITY = object()
        runner = unittest2.TextTestRunner(STREAM, DESCRIPTIONS, VERBOSITY,
                                         resultclass=MockResultClass)
        self.assertEqual(runner.resultclass, MockResultClass)

        expectedresult = (runner.stream, DESCRIPTIONS, VERBOSITY)
        self.assertEqual(runner._makeResult(), expectedresult)


    def test_oldresult(self):
        class Test(unittest2.TestCase):
            def testFoo(self):
                pass
        runner = unittest2.TextTestRunner(resultclass=OldTestResult,
                                          stream=StringIO())
        # This will raise an exception if TextTestRunner can't handle old
        # test result objects
        runner.run(Test('testFoo'))


if __name__ == '__main__':
    unittest2.main()