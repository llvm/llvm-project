from lldbsuite.test import lldbinline
from lldbsuite.test import decorators

lldbinline.MakeInlineTest(
    __file__,
    globals(),
    [
        lldbinline.expectedFailureAll(
            oslist=["windows"],
            debug_info=decorators.no_match(["pdb"]),
            bugnumber="llvm.org/pr43707",
        )
    ],
)
