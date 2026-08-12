from lldbsuite.test import lldbinline
from lldbsuite.test import decorators

lldbinline.MakeInlineTest(
    __file__,
    globals(),
    [
        decorators.requireNotWasm,
        decorators.expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24764"),
    ],
)
