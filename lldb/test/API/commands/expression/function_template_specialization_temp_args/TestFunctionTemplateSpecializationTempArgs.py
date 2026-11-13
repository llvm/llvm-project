import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestFunctionTemplateSpecializationTempArgs(TestBase):
    @skipIf(oslist=["windows"], archs=["aarch64"])
    def test_function_template_specialization_temp_args(self):
        self.build()

        (self.target, self.process, _, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp", False)
        )

        # With PDB, p0 has the type that the VType typedef names.
        vtype = "M<int>" if self.getDebugInfo() == "pdb" else "VType"
        self.expect_expr("p0", result_type=vtype, result_children=[])
