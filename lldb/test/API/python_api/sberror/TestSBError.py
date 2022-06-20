"""Test the SBError APIs."""

from lldbsuite.test.lldbtest import *

class TestSBError(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_generic_error(self):
        error = lldb.SBError()
        error.SetErrorToGenericError()
        self.assertEqual(error.GetType(), lldb.eErrorTypeGeneric)
