import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestNamespaceLocalVarSameNameObjC(TestBase):

    @add_test_categories(["gmodules", "objc"])
    def test_namespace_local_var_same_name_obj_c(self):
        self.build()

        (self.target, self.process, _, bkpt) = lldbutil.run_to_source_breakpoint(self, '// break here',
                lldb.SBFileSpec("util.mm", False))

        self.expect("expr error",
                substrs=['(NSError *) $0 ='])

        lldbutil.continue_to_breakpoint(self.process, bkpt)

        self.expect("expr error",
                substrs=['(NSError *) $1 ='])
