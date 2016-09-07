from cStringIO import StringIO

import unittest
import unittest2

from unittest2.test.support import resultFactory


class TestUnittest(unittest2.TestCase):

    def assertIsSubclass(self, actual, klass):
        self.assertTrue(issubclass(actual, klass), "Not a subclass.")

    def testInheritance(self):
        self.assertIsSubclass(unittest2.TestCase, unittest.TestCase)
        self.assertIsSubclass(unittest2.TestResult, unittest.TestResult)
        self.assertIsSubclass(unittest2.TestSuite, unittest.TestSuite)
        self.assertIsSubclass(
            unittest2.TextTestRunner,
            unittest.TextTestRunner)
        self.assertIsSubclass(unittest2.TestLoader, unittest.TestLoader)
        self.assertIsSubclass(unittest2.TextTestResult, unittest.TestResult)

    def test_new_runner_old_case(self):
        runner = unittest2.TextTestRunner(resultclass=resultFactory,
                                          stream=StringIO())

        class Test(unittest.TestCase):

            def testOne(self):
                pass
        suite = unittest2.TestSuite((Test('testOne'),))
        result = runner.run(suite)
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(len(result.errors), 0)

    def test_old_runner_new_case(self):
        runner = unittest.TextTestRunner(stream=StringIO())

        class Test(unittest2.TestCase):

            def testOne(self):
                self.assertDictEqual({}, {})

        suite = unittest.TestSuite((Test('testOne'),))
        result = runner.run(suite)
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(len(result.errors), 0)


if __name__ == '__main__':
    unittest2.main()
