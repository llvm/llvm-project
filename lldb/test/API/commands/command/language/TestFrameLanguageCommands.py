import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestCase(TestBase):
    def test(self):
        self.build()
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("lib.cpp")
        )

        frame = thread.selected_frame
        self.assertEqual(frame.GuessLanguage(), lldb.eLanguageTypeC_plus_plus_11)
        self.assertEqual(frame.name, "f()")
        self.expect(
            "help demangle",
            substrs=[
                "Demangle a C++ mangled name.",
                "Syntax: language cplusplus demangle [<mangled-name> ...]",
            ],
        )
        self.expect("demangle _Z1fv", startstr="_Z1fv ---> f()")

        # Switch the objc caller.
        self.runCmd("up")
        frame = thread.selected_frame
        self.assertEqual(frame.GuessLanguage(), lldb.eLanguageTypeObjC_plus_plus)
        self.assertEqual(frame.name, "main")
        self.expect("help demangle", error=True)
        self.expect(
            "help tagged-pointer",
            substrs=[
                "Commands for operating on Objective-C tagged pointers.",
                "Syntax: class-table <subcommand> [<subcommand-options>]",
            ],
        )
        self.expect(
            "tagged-pointer info 0",
            error=True,
            startstr="error: could not convert '0' to a valid address",
        )
