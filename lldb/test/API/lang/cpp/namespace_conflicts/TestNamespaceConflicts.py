from lldbsuite.test import lldbinline
from lldbsuite.test import decorators

lldbinline.MakeInlineTest(
    __file__,
    globals(),
    [decorators.requireExpressionEvaluation],
    test_with_pdb_debug_info=True,
)
