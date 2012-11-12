import unittest
import Test

"""
TestCase adaptor for providing a 'unittest' compatible interface to 'lit' tests.
"""

class UnresolvedError(RuntimeError):
    pass
        
class LitTestCase(unittest.TestCase):
    def __init__(self, test, lit_config):
        unittest.TestCase.__init__(self)
        self._test = test
        self._lit_config = lit_config

    def id(self):
        return self._test.getFullName()

    def shortDescription(self):
        return self._test.getFullName()

    def runTest(self):
        tr, output = self._test.config.test_format.execute(
            self._test, self._lit_config)

        if tr is Test.UNRESOLVED:
            raise UnresolvedError(output)
        elif tr.isFailure:
            self.fail(output)
