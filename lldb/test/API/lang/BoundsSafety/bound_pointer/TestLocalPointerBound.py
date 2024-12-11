import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestLocalPointerBound(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test(self):
        self.build()
        (_, self.process, _, bkpt) = lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.c"))

        self.expect_expr("foo(5)", result_type="int *")
        self.expect_expr("ptr1", result_type="int *__bidi_indexable")
        self.expect_expr("ptr2", result_type="int *__bidi_indexable")
        self.expect_expr("ps", result_type="S *__bidi_indexable")

        self.expect( "expr ptr1",
            substrs=['(int *__bidi_indexable)',
                    '$4 =',
                    'bounds:',
                    '..'])

        self.expect( "expr ptr2",
            substrs=['(int *__bidi_indexable)',
                    '$5 =',
                    'bounds:',
                    '..'])

        self.expect( "expr ps",
            substrs=['(S *__bidi_indexable)',
                    '$6 =',
                    'bounds:',
                    '..'])

        lldbutil.continue_to_breakpoint(self.process, bkpt);

        self.expect_expr("p", result_type="int *")
