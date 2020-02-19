import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestFunctionTemplateSpecializationTempArgs(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_function_template_specialization_temp_args(self):
        self.build()

        (self.target, self.process, _, bkpt) = lldbutil.run_to_source_breakpoint(self, '// break here',
                lldb.SBFileSpec("main.cpp", False))

        self.expect("expr p0",
                substrs=['(VType) $0 = {}'])
