"""
Test the --wants-dereference flag of 'type synth add'.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TypeSynthWantsDerefTestCase(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.addTearDownHook(lambda: self.runCmd("type synth clear", check=False))

    def _setup_synthetic(self, wants_deref: bool):
        self.runCmd("command script import provider.py")
        self.runCmd(
            f"type synth add -l provider.WrapperSynthProvider --wants-dereference {wants_deref} Wrapper"
        )

    def test_wants_deref_on_pointer(self):
        """With --wants-dereference true on pointer."""
        self.build()
        lldbutil.run_to_source_breakpoint(self, "return 0", lldb.SBFileSpec("main.cpp"))
        self._setup_synthetic(True)

        self.expect_var_path("wp", children=[ValueCheck(name="sum", value="30")])

    def test_wants_deref_on_reference(self):
        """With --wants-dereference true on reference."""
        self.build()
        lldbutil.run_to_source_breakpoint(self, "return 0", lldb.SBFileSpec("main.cpp"))
        self._setup_synthetic(True)

        self.expect_var_path("wr", children=[ValueCheck(name="sum", value="30")])

    def test_no_wants_deref_on_pointer(self):
        """With --wants-dereference false on pointer."""
        self.build()
        lldbutil.run_to_source_breakpoint(self, "return 0", lldb.SBFileSpec("main.cpp"))
        self._setup_synthetic(False)

        self.expect("frame variable wp", matching=False, substrs=["sum"])

    def test_no_wants_deref_on_reference(self):
        """With --wants-dereference false on reference."""
        self.build()
        lldbutil.run_to_source_breakpoint(self, "return 0", lldb.SBFileSpec("main.cpp"))
        self._setup_synthetic(False)

        self.expect("frame variable wr", matching=False, substrs=["sum"])
