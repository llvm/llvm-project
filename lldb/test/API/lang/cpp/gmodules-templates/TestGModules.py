import lldbsuite.test.lldbinline as lldbinline
from lldbsuite.test.decorators import *

lldbinline.MakeInlineTest(__file__, globals(), [
    expectedFailureAll(oslist=["linux"], bugnumber="llvm.org/pr36107",
        debug_info="gmodules")])
