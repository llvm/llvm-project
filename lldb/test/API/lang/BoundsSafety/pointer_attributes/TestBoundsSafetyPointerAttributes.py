import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestBoundsSafetyPointerAttributes(TestBase):

    mydir = TestBase.compute_mydir(__file__)


    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.c"))

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE, substrs=[' resolved, hit count = 1'])

        # Check if pointer types are displayed as expected.
        self.expect_expr("biPtr",   result_type="S *__bidi_indexable")
        self.expect_expr("iPtr",    result_type="S *__indexable")
        self.expect_expr("sPtr",    result_type="S *")
        self.expect_expr("uiPtr",   result_type="S *")
        self.expect_expr("uPtr",    result_type="S *__bidi_indexable")

        self.expect("expr biPtr",
                    substrs=['(S *__bidi_indexable)',
                    'bounds:',
                    '..'])
        self.expect("expr iPtr",
                    substrs=['(S *__indexable)',
                    'upper bound:'])
        self.expect("expr uPtr",
                    substrs=['(S *__bidi_indexable)',
                    'bounds:',
                    '..'])
        # Check if member access via attributed pointers works.
        self.expect("expr biPtr->vptr", VARIABLES_DISPLAYED_CORRECTLY,
                    startstr='(void *)')
        self.expect("expr iPtr->vptr", VARIABLES_DISPLAYED_CORRECTLY,
                    startstr='(void *)')
        self.expect("expr sPtr->vptr", VARIABLES_DISPLAYED_CORRECTLY,
                    startstr='(void *)')
        self.expect("expr uiPtr->vptr", VARIABLES_DISPLAYED_CORRECTLY,
                    startstr='(void *)')
        self.expect("expr uPtr->vptr", VARIABLES_DISPLAYED_CORRECTLY,
                    startstr='(void *)')
        self.expect_expr("biPtr->len",  result_value='4')
        self.expect_expr("iPtr->len",   result_value='4')
        self.expect_expr("sPtr->len",   result_value='4')
        self.expect_expr("uiPtr->len",  result_value='4')
        self.expect_expr("uPtr->len",   result_value='4')

        self.expect("expr *biPtr", VARIABLES_DISPLAYED_CORRECTLY,
                    startstr='(S)')
        self.expect("expr *iPtr", VARIABLES_DISPLAYED_CORRECTLY,
                    startstr='(S)')
        self.expect("expr *sPtr", VARIABLES_DISPLAYED_CORRECTLY,
                    startstr='(S)')
        self.expect("expr *uiPtr", VARIABLES_DISPLAYED_CORRECTLY,
                    startstr='(S)')
        self.expect("expr *uPtr", VARIABLES_DISPLAYED_CORRECTLY,
                    startstr='(S)')
