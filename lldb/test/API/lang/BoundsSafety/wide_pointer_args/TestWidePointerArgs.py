import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestWidePointerArgs(TestBase):

    mydir = TestBase.compute_mydir(__file__)


    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.c"))

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs=[' resolved, hit count = 1'])

        self.expect(
            "frame variable --show-types callback",
            VARIABLES_DISPLAYED_CORRECTLY,
            startstr='(int (*)(int *__bidi_indexable)) callback =')

    def test_call(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.c"))

        self.expect_expr("foo(&arr[1])", result_type="int", result_value="2")
        self.expect_expr("foo(ptrImplicitBidiIndex)", result_type="int", result_value="1")
        self.expect_expr("foo(ptrBidiIndex)", result_type="int", result_value="2")
        self.expect_expr("foo(ptrIndex)", result_type="int", result_value="3")
        self.expect_expr("foo(ptrSingle)", result_type="int", result_value="4")
        self.expect_expr("foo(ptrUnsafe)", result_type="int", result_value="5")
        self.expect_expr("(void*)ptrImplicitBidiIndex", result_type="void *")
        self.expect_expr("(void*)ptrBidiIndex", result_type="void *")
        self.expect_expr("(void*)ptrIndex", result_type="void *")
        self.expect_expr("(void*)ptrSingle", result_type="void *")

        self.expect_expr("ptrImplicitBidiIndex ? ptrImplicitBidiIndex : 0", result_type="int *__bidi_indexable")
        self.expect_expr("ptrImplicitBidiIndex ? ptrBidiIndex : 0", result_type="int *__bidi_indexable")
        self.expect_expr("ptrImplicitBidiIndex ? ptrIndex : 0", result_type="int *__indexable")
        self.expect_expr("!ptrImplicitBidiIndex ? ptrImplicitBidiIndex : arr", result_type="int *")
        self.expect_expr("ptrImplicitBidiIndex ? arr : ptrBidiIndex", result_type="int *")
        self.expect_expr("ptrImplicitBidiIndex ? ptrIndex : &arr[0]", result_type="int *")
        self.expect_expr("ptrImplicitBidiIndex ? ptrBidiIndex : ptrSingle", result_type="int *__bidi_indexable")
        self.expect_expr("!ptrImplicitBidiIndex ? ptrBidiIndex : ptrUnsafe", result_type="int *__bidi_indexable")
        self.expect_expr("ptrImplicitBidiIndex ? ptrImplicitBidiIndex : ptrIndex", result_type="int *__bidi_indexable")
        self.expect_expr("ptrImplicitBidiIndex ? ptrIndex : ptrBidiIndex", result_type="int *__indexable")
        self.expect_expr("ptrImplicitBidiIndex ? ptrIndex : ptrIndex", result_type="int *__indexable")
        self.expect_expr("ptrImplicitBidiIndex ? ptrBidiIndex : ptrIndex", result_type="int *__bidi_indexable")

        self.expect_expr("ptrIndex ? ptrImplicitBidiIndex : 0", result_type="int *__bidi_indexable")
        self.expect_expr("ptrIndex ? ptrBidiIndex : 0", result_type="int *__bidi_indexable")
        self.expect_expr("ptrBidiIndex ? ptrIndex : 0", result_type="int *__indexable")
        self.expect_expr("ptrBidiIndex ? ptrImplicitBidiIndex : arr", result_type="int *")
        self.expect_expr("ptrIndex ? &arr[0] : ptrBidiIndex", result_type="int *")
        self.expect_expr("ptrBidiIndex ? ptrIndex : ptrSingle", result_type="int *__indexable")
        self.expect_expr("ptrIndex ? ptrBidiIndex : arr", result_type="int *")
        self.expect_expr("ptrBidiIndex ? ptrImplicitBidiIndex : ptrIndex", result_type="int *__bidi_indexable")
        self.expect_expr("ptrBidiIndex ? ptrIndex : ptrBidiIndex", result_type="int *__indexable")
        self.expect_expr("ptrIndex ? ptrIndex : ptrIndex")
        self.expect_expr("!ptrIndex ? ptrBidiIndex : ptrIndex", result_type="int *__bidi_indexable")

        self.expect_expr("ptrBidiIndex ?: 0", result_type="int *__bidi_indexable")
        self.expect_expr("ptrIndex ?: 0", result_type="int *__indexable")

        self.expect_expr("callback(&arr[0])", result_type="int", result_value="1")
        self.expect_expr("callback_s(ptrImplicitBidiIndex)", result_type="int", result_value="1")
        self.expect_expr("callback_s(ptrBidiIndex)", result_type="int", result_value="2")
        self.expect_expr("callback_s(ptrIndex)", result_type="int", result_value="3")
        self.expect_expr("callback_s(ptrSingle)", result_type="int", result_value="4")
        self.expect_expr("callback_s(ptrUnsafe)", result_type="int", result_value="5")
        self.expect_expr("callback_s(arr)", result_type="int", result_value="1")
        # We still have a correctness issue here that assignments to a wide pointer only changes the pointer value.
        self.expect_expr("ptrImplicitBidiIndex = baz(ptrBidiIndex)", result_type="int *__bidi_indexable")
        self.expect_expr("ptrBidiIndex = baz(ptrSingle)", result_type="int *__bidi_indexable")
        self.expect_expr("ptrIndex = ptrImplicitBidiIndex", result_type="int *__indexable")
