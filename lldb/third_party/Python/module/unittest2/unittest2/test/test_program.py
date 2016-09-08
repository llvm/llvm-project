from cStringIO import StringIO

import sys
import unittest2

hasInstallHandler = hasattr(unittest2, 'installHandler')


class Test_TestProgram(unittest2.TestCase):

    # Horrible white box test
    def testNoExit(self):
        result = object()
        test = object()

        class FakeRunner(object):

            def run(self, test):
                self.test = test
                return result

        runner = FakeRunner()

        oldParseArgs = unittest2.TestProgram.parseArgs

        def restoreParseArgs():
            unittest2.TestProgram.parseArgs = oldParseArgs
        unittest2.TestProgram.parseArgs = lambda *args: None
        self.addCleanup(restoreParseArgs)

        def removeTest():
            del unittest2.TestProgram.test
        unittest2.TestProgram.test = test
        self.addCleanup(removeTest)

        program = unittest2.TestProgram(
            testRunner=runner, exit=False, verbosity=2)

        self.assertEqual(program.result, result)
        self.assertEqual(runner.test, test)
        self.assertEqual(program.verbosity, 2)

    class FooBar(unittest2.TestCase):

        def testPass(self):
            assert True

        def testFail(self):
            assert False

    class FooBarLoader(unittest2.TestLoader):
        """Test loader that returns a suite containing FooBar."""

        def loadTestsFromModule(self, module):
            return self.suiteClass(
                [self.loadTestsFromTestCase(Test_TestProgram.FooBar)])

    def test_NonExit(self):
        program = unittest2.main(
            exit=False,
            argv=["foobar"],
            testRunner=unittest2.TextTestRunner(
                stream=StringIO()),
            testLoader=self.FooBarLoader())
        self.assertTrue(hasattr(program, 'result'))

    def test_Exit(self):
        self.assertRaises(
            SystemExit,
            unittest2.main,
            argv=["foobar"],
            testRunner=unittest2.TextTestRunner(stream=StringIO()),
            exit=True,
            testLoader=self.FooBarLoader())

    def test_ExitAsDefault(self):
        self.assertRaises(
            SystemExit,
            unittest2.main,
            argv=["foobar"],
            testRunner=unittest2.TextTestRunner(stream=StringIO()),
            testLoader=self.FooBarLoader())


class InitialisableProgram(unittest2.TestProgram):
    exit = False
    result = None
    verbosity = 1
    defaultTest = None
    testRunner = None
    testLoader = unittest2.defaultTestLoader
    progName = 'test'
    test = 'test'

    def __init__(self, *args):
        pass

RESULT = object()


class FakeRunner(object):
    initArgs = None
    test = None
    raiseError = False

    def __init__(self, **kwargs):
        FakeRunner.initArgs = kwargs
        if FakeRunner.raiseError:
            FakeRunner.raiseError = False
            raise TypeError

    def run(self, test):
        FakeRunner.test = test
        return RESULT


class TestCommandLineArgs(unittest2.TestCase):

    def setUp(self):
        self.program = InitialisableProgram()
        self.program.createTests = lambda: None
        FakeRunner.initArgs = None
        FakeRunner.test = None
        FakeRunner.raiseError = False

    def testHelpAndUnknown(self):
        program = self.program

        def usageExit(msg=None):
            program.msg = msg
            program.exit = True
        program.usageExit = usageExit

        for opt in '-h', '-H', '--help':
            program.exit = False
            program.parseArgs([None, opt])
            self.assertTrue(program.exit)
            self.assertIsNone(program.msg)

        program.parseArgs([None, '-$'])
        self.assertTrue(program.exit)
        self.assertIsNotNone(program.msg)

    def testVerbosity(self):
        program = self.program

        for opt in '-q', '--quiet':
            program.verbosity = 1
            program.parseArgs([None, opt])
            self.assertEqual(program.verbosity, 0)

        for opt in '-v', '--verbose':
            program.verbosity = 1
            program.parseArgs([None, opt])
            self.assertEqual(program.verbosity, 2)

    def testBufferCatchFailfast(self):
        program = self.program
        for arg, attr in (('buffer', 'buffer'), ('failfast', 'failfast'),
                          ('catch', 'catchbreak')):
            if attr == 'catch' and not hasInstallHandler:
                continue

            short_opt = '-%s' % arg[0]
            long_opt = '--%s' % arg
            for opt in short_opt, long_opt:
                setattr(program, attr, None)

                program.parseArgs([None, opt])
                self.assertTrue(getattr(program, attr))

            for opt in short_opt, long_opt:
                not_none = object()
                setattr(program, attr, not_none)

                program.parseArgs([None, opt])
                self.assertEqual(getattr(program, attr), not_none)

    def testRunTestsRunnerClass(self):
        program = self.program

        program.testRunner = FakeRunner
        program.verbosity = 'verbosity'
        program.failfast = 'failfast'
        program.buffer = 'buffer'

        program.runTests()

        self.assertEqual(FakeRunner.initArgs, {'verbosity': 'verbosity',
                                               'failfast': 'failfast',
                                               'buffer': 'buffer'})
        self.assertEqual(FakeRunner.test, 'test')
        self.assertIs(program.result, RESULT)

    def testRunTestsRunnerInstance(self):
        program = self.program

        program.testRunner = FakeRunner()
        FakeRunner.initArgs = None

        program.runTests()

        # A new FakeRunner should not have been instantiated
        self.assertIsNone(FakeRunner.initArgs)

        self.assertEqual(FakeRunner.test, 'test')
        self.assertIs(program.result, RESULT)

    def testRunTestsOldRunnerClass(self):
        program = self.program

        FakeRunner.raiseError = True
        program.testRunner = FakeRunner
        program.verbosity = 'verbosity'
        program.failfast = 'failfast'
        program.buffer = 'buffer'
        program.test = 'test'

        program.runTests()

        # If initialising raises a type error it should be retried
        # without the new keyword arguments
        self.assertEqual(FakeRunner.initArgs, {})
        self.assertEqual(FakeRunner.test, 'test')
        self.assertIs(program.result, RESULT)

    def testCatchBreakInstallsHandler(self):
        module = sys.modules['unittest2.main']
        original = module.installHandler

        def restore():
            module.installHandler = original
        self.addCleanup(restore)

        self.installed = False

        def fakeInstallHandler():
            self.installed = True
        module.installHandler = fakeInstallHandler

        program = self.program
        program.catchbreak = True

        program.testRunner = FakeRunner

        program.runTests()
        self.assertTrue(self.installed)


if __name__ == '__main__':
    unittest2.main()
