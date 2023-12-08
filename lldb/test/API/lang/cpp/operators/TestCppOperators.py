from lldbsuite.test import lldbinline
from lldbsuite.test import decorators

lldbinline.MakeInlineTest(
    __file__,
    globals(),
    [decorators.expectedFailureAll(bugnumber="llvm.org/pr50814", compiler="gcc")],
)
