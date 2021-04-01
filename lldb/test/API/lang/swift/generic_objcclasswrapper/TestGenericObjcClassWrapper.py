import lldbsuite.test.lldbinline as lldbinline
from lldbsuite.test.decorators import *

lldbinline.MakeInlineTest(__file__, globals(), decorators=[swiftTest, skipUnlessDarwin,
    skipIf(bugnumber='rdar://76105456', oslist=['macosx'])])

