"""Test that lldb picks the correct DWARF location list entry with a return-pc out of bounds."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LocationListLookupTestCase(TestBase):
    def launch(self) -> lldb.SBProcess:
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        self.dbg.SetAsync(False)

        li = lldb.SBLaunchInfo(["a.out"])
        error = lldb.SBError()
        process = target.Launch(li, error)
        self.assertTrue(process.IsValid())
        self.assertTrue(process.is_stopped)

        return process

    def check_local_vars(self, process: lldb.SBProcess, check_expr: bool):
        # Find `bar` on the stack, then
        # make sure we can read out the local
        # variables (with both `frame var` and `expr`)
        for f in process.selected_thread.frames:
            frame_name = f.GetDisplayFunctionName()
            if frame_name is not None and frame_name.startswith("Foo::bar"):
                argv = f.GetValueForVariablePath("argv").GetChildAtIndex(0)
                strm = lldb.SBStream()
                argv.GetDescription(strm)
                self.assertNotEqual(strm.GetData().find("a.out"), -1)

                if check_expr:
                    process.selected_thread.selected_frame = f
                    self.expect_expr("this", result_type="Foo *")

    @skipIf(oslist=["linux"], archs=["arm"])
    @skipIfDarwin
    def test_loclist_frame_var(self):
        self.build()
        self.check_local_vars(self.launch(), check_expr=False)

    @skipIf(dwarf_version=["<", "3"])
    @skipIf(compiler="clang", compiler_version=["<", "12.0"])
    @skipUnlessDarwin
    def test_loclist_expr(self):
        self.build()
        self.check_local_vars(self.launch(), check_expr=True)
