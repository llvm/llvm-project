import sys
import warnings

import unittest2


def resultFactory(*_):
    return unittest2.TestResult()

class OldTestResult(object):
    """An object honouring TestResult before startTestRun/stopTestRun."""

    def __init__(self, *_):
        self.failures = []
        self.errors = []
        self.testsRun = 0
        self.shouldStop = False

    def startTest(self, test):
        pass

    def stopTest(self, test):
        pass

    def addError(self, test, err):
        self.errors.append((test, err))

    def addFailure(self, test, err):
        self.failures.append((test, err))

    def addSuccess(self, test):
        pass

    def wasSuccessful(self):
        return True

    def printErrors(self):
        pass

class LoggingResult(unittest2.TestResult):
    def __init__(self, log):
        self._events = log
        super(LoggingResult, self).__init__()

    def startTest(self, test):
        self._events.append('startTest')
        super(LoggingResult, self).startTest(test)

    def startTestRun(self):
        self._events.append('startTestRun')
        super(LoggingResult, self).startTestRun()

    def stopTest(self, test):
        self._events.append('stopTest')
        super(LoggingResult, self).stopTest(test)

    def stopTestRun(self):
        self._events.append('stopTestRun')
        super(LoggingResult, self).stopTestRun()

    def addFailure(self, *args):
        self._events.append('addFailure')
        super(LoggingResult, self).addFailure(*args)

    def addSuccess(self, *args):
        self._events.append('addSuccess')
        super(LoggingResult, self).addSuccess(*args)

    def addError(self, *args):
        self._events.append('addError')
        super(LoggingResult, self).addError(*args)

    def addSkip(self, *args):
        self._events.append('addSkip')
        super(LoggingResult, self).addSkip(*args)

    def addExpectedFailure(self, *args):
        self._events.append('addExpectedFailure')
        super(LoggingResult, self).addExpectedFailure(*args)

    def addUnexpectedSuccess(self, *args):
        self._events.append('addUnexpectedSuccess')
        super(LoggingResult, self).addUnexpectedSuccess(*args)


class EqualityMixin(object):
    """Used as a mixin for TestCase"""

    # Check for a valid __eq__ implementation
    def test_eq(self):
        for obj_1, obj_2 in self.eq_pairs:
            self.assertEqual(obj_1, obj_2)
            self.assertEqual(obj_2, obj_1)

    # Check for a valid __ne__ implementation
    def test_ne(self):
        for obj_1, obj_2 in self.ne_pairs:
            self.assertNotEqual(obj_1, obj_2)
            self.assertNotEqual(obj_2, obj_1)

class HashingMixin(object):
    """Used as a mixin for TestCase"""

    # Check for a valid __hash__ implementation
    def test_hash(self):
        for obj_1, obj_2 in self.eq_pairs:
            try:
                if not hash(obj_1) == hash(obj_2):
                    self.fail("%r and %r do not hash equal" % (obj_1, obj_2))
            except KeyboardInterrupt:
                raise
            except Exception, e:
                self.fail("Problem hashing %r and %r: %s" % (obj_1, obj_2, e))

        for obj_1, obj_2 in self.ne_pairs:
            try:
                if hash(obj_1) == hash(obj_2):
                    self.fail("%s and %s hash equal, but shouldn't" %
                              (obj_1, obj_2))
            except KeyboardInterrupt:
                raise
            except Exception, e:
                self.fail("Problem hashing %s and %s: %s" % (obj_1, obj_2, e))



# copied from Python 2.6
try:
    from warnings import catch_warnings
except ImportError:
    class catch_warnings(object):
        def __init__(self, record=False, module=None):
            self._record = record
            self._module = sys.modules['warnings']
            self._entered = False
    
        def __repr__(self):
            args = []
            if self._record:
                args.append("record=True")
            name = type(self).__name__
            return "%s(%s)" % (name, ", ".join(args))
    
        def __enter__(self):
            if self._entered:
                raise RuntimeError("Cannot enter %r twice" % self)
            self._entered = True
            self._filters = self._module.filters
            self._module.filters = self._filters[:]
            self._showwarning = self._module.showwarning
            if self._record:
                log = []
                def showwarning(*args, **kwargs):
                    log.append(WarningMessage(*args, **kwargs))
                self._module.showwarning = showwarning
                return log
            else:
                return None
    
        def __exit__(self, *exc_info):
            if not self._entered:
                raise RuntimeError("Cannot exit %r without entering first" % self)
            self._module.filters = self._filters
            self._module.showwarning = self._showwarning

    class WarningMessage(object):
        _WARNING_DETAILS = ("message", "category", "filename", "lineno", "file",
                            "line")
        def __init__(self, message, category, filename, lineno, file=None,
                        line=None):
            local_values = locals()
            for attr in self._WARNING_DETAILS:
                setattr(self, attr, local_values[attr])
            self._category_name = None
            if category.__name__:
                self._category_name = category.__name__

