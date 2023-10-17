from lldbsuite.test import lldbinline
from lldbsuite.test import decorators

decor = [decorators.skipIf(compiler="clang", compiler_version=["<", "16.0"])]
lldbinline.MakeInlineTest(__file__, globals(), decor)
