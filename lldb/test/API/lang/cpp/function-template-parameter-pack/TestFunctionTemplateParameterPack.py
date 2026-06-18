from lldbsuite.test import lldbinline
from lldbsuite.test import decorators

# https://bugs.llvm.org/show_bug.cgi?id=35920
# This test stresses expression evaluation support for template functions.
# This is currently not fully supported. Hence XFAIL this test.
lldbinline.MakeInlineTest(__file__, globals(), [decorators.expectedFailureAll])
