from lldbsuite.test import lldbinline
from lldbsuite.test import decorators

lldbinline.MakeInlineTest(
    __file__,
    globals(),
    [lldbinline.expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr43707")],
)
