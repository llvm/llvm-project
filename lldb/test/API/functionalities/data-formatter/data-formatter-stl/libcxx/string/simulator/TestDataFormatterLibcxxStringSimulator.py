"""
Test we can understand various layouts of the libc++'s std::string
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LibcxxStringDataFormatterSimulatorTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def _run_test(self, defines):
        cxxflags_extras = " ".join(["-D%s" % d for d in defines])
        self.build(dictionary=dict(CXXFLAGS_EXTRAS=cxxflags_extras))
        lldbutil.run_to_source_breakpoint(self, '// Break here',
                lldb.SBFileSpec("main.cpp"))
        self.expect_var_path("shortstring", summary='"short"')
        self.expect_var_path("longstring", summary='"I am a very long string"')

    def test_v1_layout(self):
        """ Current v1 layout. """
        self._run_test([])

    def test_v2_layout(self):
        """ Current v2 layout. """
        self._run_test(["ALTERNATE_LAYOUT"])

    def test_v1_layout_bitmasks(self):
        """ Pre-D123580 v1 layout. """
        self._run_test(["BITMASKS"])

    def test_v2_layout_bitmasks(self):
        """ Pre-D123580 v2 layout. """
        self._run_test(["ALTERNATE_LAYOUT", "BITMASKS"])

    def test_v2_layout_subclass_padding(self):
        """ Pre-c3d0205ee771 v2 layout. """
        self._run_test(["ALTERNATE_LAYOUT", "BITMASKS", "SUBCLASS_PADDING"])

