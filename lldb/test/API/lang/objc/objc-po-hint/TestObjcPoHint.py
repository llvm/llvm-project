import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestObjcPoHint(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_show_po_hint(self):
        ### Test that the po hint is shown once with the DWIM print command
        self.build()
        _, _, _, _ = lldbutil.run_to_source_breakpoint(
            self, "Set breakpoint here", lldb.SBFileSpec("main.m")
        )
        # Make sure the hint is printed the first time
        self.expect(
            "dwim-print -O -- foo",
            substrs=[
                "note: object description requested, but type doesn't implement "
                'a custom object description. Consider using "p" instead of "po"',
                "<Foo: 0x",
            ],
        )

        # Make sure it's not printed again.
        self.expect(
            "dwim-print -O -- foo",
            substrs=["note: object description"],
            matching=False,
        )

    def test_show_po_hint_disabled(self):
        ### Test that when the setting is disabled the hint is not printed
        self.build()
        _, _, _, _ = lldbutil.run_to_source_breakpoint(
            self, "Set breakpoint here", lldb.SBFileSpec("main.m")
        )
        self.runCmd("setting set show-dont-use-po-hint false")
        # Make sure the hint is printed the first time
        self.expect(
            "dwim-print -O -- foo",
            substrs=["note: object description"],
            matching=False,
        )
