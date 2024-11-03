import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestObjCXXHideRuntimeSupportValues(TestBase):
    def test_hide_runtime_support_values(self):
        self.build()
        _, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.mm")
        )

        var_opts = lldb.SBVariablesOptions()
        var_opts.SetIncludeArguments(True)
        var_opts.SetIncludeLocals(True)
        var_opts.SetInScopeOnly(True)
        var_opts.SetIncludeStatics(False)
        var_opts.SetIncludeRuntimeSupportValues(False)
        var_opts.SetUseDynamic(lldb.eDynamicCanRunTarget)
        values = self.frame().GetVariables(var_opts)

        def shows_var(name):
            for value in values:
                if value.name == name:
                    return True
            return False

        # ObjC method.
        values = self.frame().GetVariables(var_opts)
        self.assertFalse(shows_var("this"))
        self.assertTrue(shows_var("self"))
        self.assertTrue(shows_var("_cmd"))
        self.assertTrue(shows_var("c"))

        process.Continue()
        # C++ method.
        values = self.frame().GetVariables(var_opts)
        self.assertTrue(shows_var("this"))
        self.assertFalse(shows_var("self"))
        self.assertFalse(shows_var("_cmd"))
