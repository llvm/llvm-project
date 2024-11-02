import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestVLA(TestBase):

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

        def test(a):
            children = []
            for i in range(a):
                name = "[%d]"%i
                value = str(a-i)
                self.expect_var_path("vla"+name, type="int", value=value)
                self.expect_expr("vla"+name, result_type="int",
                        result_value=value)
                children.append(ValueCheck(name=name, value=value))
            self.expect_var_path("vla", type="int[]", children=children)
            self.expect("expr vla", error=True, substrs=["incomplete"])

        test(2)
        process.Continue()
        test(4)

