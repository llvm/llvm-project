"""
Test thread step-in ignores frames according to the "Avoid regexp" option.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class StepAvoidsRegexTestCase(TestBase):
    def hit_correct_function(self, pattern):
        name = self.thread.frames[0].GetFunctionName()
        self.assertTrue(
            pattern in name,
            "Got to '%s' not the expected function '%s'." % (name, pattern),
        )

    def setUp(self):
        TestBase.setUp(self)
        self.dbg.HandleCommand(
            "settings set target.process.thread.step-avoid-regexp ^ignore::"
        )

    @skipIfWindows
    @skipIf(compiler="clang", compiler_version=["<", "11.0"])
    def test_step_avoid_regex(self):
        """Tests stepping into a function which matches the avoid regex"""
        self.build()
        (_, _, self.thread, _) = lldbutil.run_to_source_breakpoint(
            self, "main", lldb.SBFileSpec("main.cpp")
        )

        # Try to step into ignore::auto_ret
        self.thread.StepInto()
        self.hit_correct_function("main")

        # Try to step into ignore::with_tag
        self.thread.StepInto()
        self.hit_correct_function("main")

        # Try to step into ignore::decltype_auto_ret
        self.thread.StepInto()
        self.hit_correct_function("main")

        # Try to step into ignore::with_tag_template
        self.thread.StepInto()
        self.hit_correct_function("main")

        # Step into with_tag_template_returns_ignore which is outside the 'ignore::'
        # namespace but returns a type from 'ignore::'
        self.thread.StepInto()
        self.hit_correct_function("with_tag_template_returns_ignore")
