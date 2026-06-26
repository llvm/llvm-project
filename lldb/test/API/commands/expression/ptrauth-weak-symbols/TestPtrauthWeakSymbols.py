"""
Test that we can compile expressions referring to absent weak symbols from a dylib on arm64e.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbarm64e import Arm64eTestBase


class TestWeakSymbolsInExpressionsOnArm64e(Arm64eTestBase):
    SHARED_BUILD_TESTCASE = False
    NO_DEBUG_INFO_TESTCASE = True

    @skipIf(compiler="clang", compiler_version=["<", "19.0"])
    def test_weak_symbol_in_expr(self):
        self.build()

        hidden_dir = os.path.join(self.getBuildDir(), "hidden")
        hidden_dylib = os.path.join(hidden_dir, "libdylib.dylib")

        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetWorkingDirectory(self.getBuildDir())
        launch_info.SetLaunchFlags(lldb.eLaunchFlagInheritTCCFromParent)

        self.target, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self,
            "Set a breakpoint here",
            lldb.SBFileSpec("main.c"),
            launch_info=launch_info,
            extra_images=[hidden_dylib],
        )

        # First import the dylib module to get the type info for the weak
        # symbol. Add the source dir to the module search path and then run
        # @import to introduce it into the expression context.
        self.dbg.HandleCommand(
            f"settings set target.clang-module-search-paths {self.getSourceDir()}"
        )

        self.frame = thread.frames[0]
        self.assertTrue(self.frame.IsValid(), "Got a good frame")
        options = lldb.SBExpressionOptions()
        options.SetLanguage(lldb.eLanguageTypeObjC)
        result = self.frame.EvaluateExpression("@import Dylib", options)

        # Now run expressions that reference absent and present weak function symbols.
        absent_expr = (
            "if (&absent_weak_function != NULL) { sink = 100; } else { sink = 0; }; 10"
        )
        present_expr = (
            "if (&present_weak_function != NULL) { sink = 100; } else { sink = 0; }; 10"
        )

        value = self.target.FindFirstGlobalVariable("sink")
        value.SetValueFromCString("0")

        result = self.frame.EvaluateExpression(absent_expr)
        self.assertSuccess(result.GetError(), "absent_weak_function expr failed")
        self.assertEqual(value.GetValueAsSigned(), 0)

        result = self.frame.EvaluateExpression(present_expr)
        self.assertSuccess(result.GetError(), "present_weak_function expr failed")
        self.assertEqual(value.GetValueAsSigned(), 100)
