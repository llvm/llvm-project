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

        # Test `help`.
        self.expect(
            "help demangle",
            substrs=[
                "Demangle a C++ mangled name.",
                "Syntax: language cplusplus demangle [<mangled-name> ...]",
            ],
        )

        # Run a `language cplusplus` command.
        self.expect(f"demangle _Z1fv", startstr="_Z1fv ---> f()")
        # Test prefix matching.
        self.expect("dem _Z1fv", startstr="_Z1fv ---> f()")

        # Select the objc caller.
        self.runCmd("up")
        frame = thread.selected_frame
        self.assertEqual(frame.GuessLanguage(), lldb.eLanguageTypeObjC_plus_plus)
        self.assertEqual(frame.name, "main")

        # Ensure `demangle` doesn't resolve from the objc frame.
        self.expect("help demangle", error=True)
        # Run a `language objc` command.
        self.expect(
            "tagged-pointer info 0",
            error=True,
            startstr="error: could not convert '0' to a valid address",
        )
