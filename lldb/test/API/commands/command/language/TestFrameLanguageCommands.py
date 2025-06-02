import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestCase(TestBase):
    @skipUnlessDarwin
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
        self.expect("demangle _Z1fv", startstr="_Z1fv ---> f()")
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
            "tagged-pointer",
            substrs=[
                "Commands for operating on Objective-C tagged pointers.",
                "Syntax: tagged-pointer <subcommand> [<subcommand-options>]",
                "The following subcommands are supported:",
                "info -- Dump information on a tagged pointer.",
            ],
        )

        # To ensure compatability with existing scripts, a language specific
        # command must not be invoked if another command (such as a python
        # command) has the language specific command name as its prefix.
        #
        # For example, this test loads a `tagged-pointer-collision` command. A
        # script could exist that invokes this command using its prefix
        # `tagged-pointer`, under the assumption that "tagged-pointer" uniquely
        # identifies the python command `tagged-pointer-collision`.
        self.runCmd("command script import commands.py")
        self.expect("tagged-pointer", startstr="ran tagged-pointer-collision")
