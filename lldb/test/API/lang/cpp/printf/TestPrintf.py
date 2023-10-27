from lldbsuite.test import lldbinline, lldbplatformutil
from lldbsuite.test import decorators

lldbinline.MakeInlineTest(
    __file__,
    globals(),
    [decorators.expectedFailureAll(bugnumber="llvm.org/PR36715", oslist=["windows"])],
)
