"""Running tests"""

import sys
import time
import unittest
import progress

from unittest2 import result

try:
    from unittest2.signals import registerResult
except ImportError:
    def registerResult(_):
        pass

__unittest = True


class _WritelnDecorator(object):
    """Used to decorate file-like objects with a handy 'writeln' method"""

    def __init__(self, stream):
        self.stream = stream

    def __getattr__(self, attr):
        if attr in ('stream', '__getstate__'):
            raise AttributeError(attr)
        return getattr(self.stream, attr)

    def writeln(self, arg=None):
        if arg:
            self.write(arg)
        self.write('\n')  # text-mode streams translate to \r\n if needed


class TextTestResult(result.TestResult):
    """A test result class that can print formatted text results to a stream.

    Used by TextTestRunner.
    """
    separator1 = '=' * 70
    separator2 = '-' * 70

    def __init__(self, stream, descriptions, verbosity):
        super(TextTestResult, self).__init__()
        self.stream = stream
        self.showAll = verbosity > 1
        self.dots = verbosity == 1
        self.descriptions = descriptions
        self.progressbar = None

        if self.dots:
            self.stream.writeln(
                ".=success F=fail E=error s=skipped x=expected-fail u=unexpected-success")
            self.stream.writeln("")
            self.stream.flush()

    def getDescription(self, test):
        doc_first_line = test.shortDescription()
        if self.descriptions and doc_first_line:
            return '\n'.join((str(test), doc_first_line))
        else:
            return str(test)

    def startTest(self, test):
        super(TextTestResult, self).startTest(test)
        if self.showAll:
            self.stream.write(self.getDescription(test))
            self.stream.write(" ... ")
            self.stream.flush()

    def newTestResult(self, test, result_short, result_long):
        if self.showAll:
            self.stream.writeln(result_long)
        elif self.progressbar:
            self.progressbar.__add__(1)
            self.progressbar.add_event(result_short)
            self.progressbar.show_progress()
        elif self.dots:
            self.stream.write(result_short)
            self.stream.flush()

    def addSuccess(self, test):
        super(TextTestResult, self).addSuccess(test)
        if self.progressbar:
            self.newTestResult(test, "ok", "ok")
        else:
            self.newTestResult(test, ".", "ok")

    def addError(self, test, err):
        super(TextTestResult, self).addError(test, err)
        self.newTestResult(test, "E", "ERROR")

    def addFailure(self, test, err):
        super(TextTestResult, self).addFailure(test, err)
        self.newTestResult(test, "F", "FAILURE")

    def addSkip(self, test, reason):
        super(TextTestResult, self).addSkip(test, reason)
        self.newTestResult(test, "s", "skipped %r" % (reason,))

    def addExpectedFailure(self, test, err, bugnumber):
        super(TextTestResult, self).addExpectedFailure(test, err, bugnumber)
        self.newTestResult(test, "x", "expected failure")

    def addUnexpectedSuccess(self, test, bugnumber):
        super(TextTestResult, self).addUnexpectedSuccess(test, bugnumber)
        self.newTestResult(test, "u", "unexpected success")

    def printErrors(self):
        if self.progressbar:
            self.progressbar.complete()
            self.progressbar.show_progress()
        if self.dots or self.showAll:
            self.stream.writeln()
        self.printErrorList('ERROR', self.errors)
        self.printErrorList('FAIL', self.failures)

    def printErrorList(self, flavour, errors):
        for test, err in errors:
            self.stream.writeln(self.separator1)
            self.stream.writeln("%s: %s" %
                                (flavour, self.getDescription(test)))
            self.stream.writeln(self.separator2)
            self.stream.writeln("%s" % err)

    def stopTestRun(self):
        super(TextTestResult, self).stopTestRun()
        self.printErrors()


class TextTestRunner(unittest.TextTestRunner):
    """A test runner class that displays results in textual form.

    It prints out the names of tests as they are run, errors as they
    occur, and a summary of the results at the end of the test run.
    """
    resultclass = TextTestResult

    def __init__(self, stream=sys.stderr, descriptions=True, verbosity=1,
                 failfast=False, buffer=False, resultclass=None):
        self.stream = _WritelnDecorator(stream)
        self.descriptions = descriptions
        self.verbosity = verbosity
        self.failfast = failfast
        self.buffer = buffer
        if resultclass is not None:
            self.resultclass = resultclass

    def _makeResult(self):
        return self.resultclass(self.stream, self.descriptions, self.verbosity)

    def run(self, test):
        "Run the given test case or test suite."
        result = self._makeResult()
        result.failfast = self.failfast
        result.buffer = self.buffer
        registerResult(result)

        startTime = time.time()
        startTestRun = getattr(result, 'startTestRun', None)
        if startTestRun is not None:
            startTestRun()
        try:
            test(result)
        finally:
            stopTestRun = getattr(result, 'stopTestRun', None)
            if stopTestRun is not None:
                stopTestRun()
            else:
                result.printErrors()
        stopTime = time.time()
        timeTaken = stopTime - startTime
        if hasattr(result, 'separator2'):
            self.stream.writeln(result.separator2)
        run = result.testsRun
        self.stream.writeln("Ran %d test%s in %.3fs" %
                            (run, run != 1 and "s" or "", timeTaken))
        self.stream.writeln()

        expectedFails = unexpectedSuccesses = skipped = passed = failed = errored = 0
        try:
            results = map(len, (result.expectedFailures,
                                result.unexpectedSuccesses,
                                result.skipped,
                                result.passes,
                                result.failures,
                                result.errors))
            expectedFails, unexpectedSuccesses, skipped, passed, failed, errored = results
        except AttributeError:
            pass
        infos = []
        infos.append("%d passes" % passed)
        infos.append("%d failures" % failed)
        infos.append("%d errors" % errored)
        infos.append("%d skipped" % skipped)
        infos.append("%d expected failures" % expectedFails)
        infos.append("%d unexpected successes" % unexpectedSuccesses)
        self.stream.write("RESULT: ")
        if not result.wasSuccessful():
            self.stream.write("FAILED")
        else:
            self.stream.write("PASSED")

        self.stream.writeln(" (%s)" % (", ".join(infos),))
        return result
