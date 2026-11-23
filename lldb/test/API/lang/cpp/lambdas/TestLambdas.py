from lldbsuite.test import lldbinline
from lldbsuite.test import decorators

lldbinline.MakeInlineTest(
    __file__,
    globals(),
    [
        decorators.expectedFailureAll(
            bugnumber="https://github.com/llvm/llvm-project/issues/149477"
        )
    ],
)
