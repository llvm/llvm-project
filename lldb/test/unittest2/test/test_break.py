import gc
import os
import weakref

from cStringIO import StringIO

try:
    import signal
except ImportError:
    signal = None

import unittest2


class TestBreak(unittest2.TestCase):
    
    def setUp(self):
        self._default_handler = signal.getsignal(signal.SIGINT)
        
    def tearDown(self):
        signal.signal(signal.SIGINT, self._default_handler)
        unittest2.signals._results = weakref.WeakKeyDictionary()
        unittest2.signals._interrupt_handler = None

        
    def testInstallHandler(self):
        default_handler = signal.getsignal(signal.SIGINT)
        unittest2.installHandler()
        self.assertNotEqual(signal.getsignal(signal.SIGINT), default_handler)
        
        try:
            pid = os.getpid()
            os.kill(pid, signal.SIGINT)
        except KeyboardInterrupt:
            self.fail("KeyboardInterrupt not handled")
            
        self.assertTrue(unittest2.signals._interrupt_handler.called)
    
    def testRegisterResult(self):
        result = unittest2.TestResult()
        unittest2.registerResult(result)
        
        for ref in unittest2.signals._results:
            if ref is result:
                break
            elif ref is not result:
                self.fail("odd object in result set")
        else:
            self.fail("result not found")
        
        
    def testInterruptCaught(self):
        default_handler = signal.getsignal(signal.SIGINT)
        
        result = unittest2.TestResult()
        unittest2.installHandler()
        unittest2.registerResult(result)
        
        self.assertNotEqual(signal.getsignal(signal.SIGINT), default_handler)
        
        def test(result):
            pid = os.getpid()
            os.kill(pid, signal.SIGINT)
            result.breakCaught = True
            self.assertTrue(result.shouldStop)
        
        try:
            test(result)
        except KeyboardInterrupt:
            self.fail("KeyboardInterrupt not handled")
        self.assertTrue(result.breakCaught)
    
    
    def testSecondInterrupt(self):
        result = unittest2.TestResult()
        unittest2.installHandler()
        unittest2.registerResult(result)
        
        def test(result):
            pid = os.getpid()
            os.kill(pid, signal.SIGINT)
            result.breakCaught = True
            self.assertTrue(result.shouldStop)
            os.kill(pid, signal.SIGINT)
            self.fail("Second KeyboardInterrupt not raised")
        
        try:
            test(result)
        except KeyboardInterrupt:
            pass
        else:
            self.fail("Second KeyboardInterrupt not raised")
        self.assertTrue(result.breakCaught)

    
    def testTwoResults(self):
        unittest2.installHandler()
        
        result = unittest2.TestResult()
        unittest2.registerResult(result)
        new_handler = signal.getsignal(signal.SIGINT)
        
        result2 = unittest2.TestResult()
        unittest2.registerResult(result2)
        self.assertEqual(signal.getsignal(signal.SIGINT), new_handler)
        
        result3 = unittest2.TestResult()
        
        def test(result):
            pid = os.getpid()
            os.kill(pid, signal.SIGINT)
        
        try:
            test(result)
        except KeyboardInterrupt:
            self.fail("KeyboardInterrupt not handled")
        
        self.assertTrue(result.shouldStop)
        self.assertTrue(result2.shouldStop)
        self.assertFalse(result3.shouldStop)
    
    
    def testHandlerReplacedButCalled(self):
        # If our handler has been replaced (is no longer installed) but is
        # called by the *new* handler, then it isn't safe to delay the
        # SIGINT and we should immediately delegate to the default handler
        unittest2.installHandler()
        
        handler = signal.getsignal(signal.SIGINT)
        def new_handler(frame, signum):
            handler(frame, signum)
        signal.signal(signal.SIGINT, new_handler)
        
        try:
            pid = os.getpid()
            os.kill(pid, signal.SIGINT)
        except KeyboardInterrupt:
            pass
        else:
            self.fail("replaced but delegated handler doesn't raise interrupt")
    
    def testRunner(self):
        # Creating a TextTestRunner with the appropriate argument should
        # register the TextTestResult it creates
        runner = unittest2.TextTestRunner(stream=StringIO())
        
        result = runner.run(unittest2.TestSuite())
        self.assertIn(result, unittest2.signals._results)
    
    def testWeakReferences(self):
        # Calling registerResult on a result should not keep it alive
        result = unittest2.TestResult()
        unittest2.registerResult(result)
        
        ref = weakref.ref(result)
        del result
        
        # For non-reference counting implementations
        gc.collect();gc.collect()
        self.assertIsNone(ref())
        
    
    def testRemoveResult(self):
        result = unittest2.TestResult()
        unittest2.registerResult(result)
        
        unittest2.installHandler()
        self.assertTrue(unittest2.removeResult(result))
        
        # Should this raise an error instead?
        self.assertFalse(unittest2.removeResult(unittest2.TestResult()))

        try:
            pid = os.getpid()
            os.kill(pid, signal.SIGINT)
        except KeyboardInterrupt:
            pass
        
        self.assertFalse(result.shouldStop)
    
    def testMainInstallsHandler(self):
        failfast = object()
        test = object()
        verbosity = object()
        result = object()
        default_handler = signal.getsignal(signal.SIGINT)

        class FakeRunner(object):
            initArgs = []
            runArgs = []
            def __init__(self, *args, **kwargs):
                self.initArgs.append((args, kwargs))
            def run(self, test):
                self.runArgs.append(test)
                return result
        
        class Program(unittest2.TestProgram):
            def __init__(self, catchbreak): 
                self.exit = False
                self.verbosity = verbosity
                self.failfast = failfast
                self.catchbreak = catchbreak
                self.testRunner = FakeRunner
                self.test = test
                self.result = None
        
        p = Program(False)
        p.runTests()
        
        self.assertEqual(FakeRunner.initArgs, [((), {'verbosity': verbosity, 
                                                'failfast': failfast,
                                                'buffer': None})])
        self.assertEqual(FakeRunner.runArgs, [test])
        self.assertEqual(p.result, result)
        
        self.assertEqual(signal.getsignal(signal.SIGINT), default_handler)
        
        FakeRunner.initArgs = []
        FakeRunner.runArgs = []
        p = Program(True)
        p.runTests()
        
        self.assertEqual(FakeRunner.initArgs, [((), {'verbosity': verbosity, 
                                                'failfast': failfast,
                                                'buffer': None})])
        self.assertEqual(FakeRunner.runArgs, [test])
        self.assertEqual(p.result, result)
        
        self.assertNotEqual(signal.getsignal(signal.SIGINT), default_handler)


    def testRemoveHandler(self):
        default_handler = signal.getsignal(signal.SIGINT)
        unittest2.installHandler()
        unittest2.removeHandler()
        self.assertEqual(signal.getsignal(signal.SIGINT), default_handler)

        # check that calling removeHandler multiple times has no ill-effect
        unittest2.removeHandler()
        self.assertEqual(signal.getsignal(signal.SIGINT), default_handler)
    
    def testRemoveHandlerAsDecorator(self):
        default_handler = signal.getsignal(signal.SIGINT)
        unittest2.installHandler()
        
        @unittest2.removeHandler
        def test():
            self.assertEqual(signal.getsignal(signal.SIGINT), default_handler)
        
        test()
        self.assertNotEqual(signal.getsignal(signal.SIGINT), default_handler)
        

# Should also skip some tests on Jython
skipper = unittest2.skipUnless(hasattr(os, 'kill') and signal is not None, 
                               "test uses os.kill(...) and the signal module")
TestBreak = skipper(TestBreak)

if __name__ == '__main__':
    unittest2.main()
