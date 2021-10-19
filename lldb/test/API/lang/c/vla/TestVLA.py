import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestVLA(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIf(compiler="clang", compiler_version=['<', '8.0'])
    def test_variable_list(self):
        self.build()
        _, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec('main.c'))

        # Make sure no helper expressions show up in frame variable.
        var_opts = lldb.SBVariablesOptions()
        var_opts.SetIncludeArguments(False)
        var_opts.SetIncludeLocals(True)
        var_opts.SetInScopeOnly(True)
        var_opts.SetIncludeStatics(False)
        var_opts.SetIncludeRuntimeSupportValues(False)
        var_opts.SetUseDynamic(lldb.eDynamicCanRunTarget)
        all_locals = self.frame().GetVariables(var_opts)
        for value in all_locals:
            self.assertNotIn("vla_expr", value.name)

    @decorators.skipIf(compiler="clang", compiler_version=['<', '8.0'])
    def test_vla(self):
        self.build()
        _, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec('main.c'))

        def test(a, array):
            for i in range(a):
                self.expect("fr v vla[%d]"%i, substrs=["int", "%d"%(a-i)])
                self.expect("expr vla[%d]"%i, substrs=["int", "%d"%(a-i)])
            self.expect("fr v vla", substrs=array)
            self.expect("expr vla", error=True, substrs=["incomplete"])

        test(2, ["int []", "[0] = 2, [1] = 1"])
        process.Continue()
        test(4, ["int []", "[0] = 4, [1] = 3, [2] = 2, [3] = 1"])

