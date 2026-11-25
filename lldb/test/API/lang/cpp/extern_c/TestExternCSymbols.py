from lldbsuite.test import lldbinline
from lldbsuite.test import decorators

lldbinline.MakeInlineTest(
    __file__,
    globals(),
    decorators=[
        decorators.skipIfWasm,  # no expression evaluation
        decorators.expectedFailureAll(
            debug_info=["pdb"], bugnumber="llvm.org/pr149498"
        ),
    ],
)
