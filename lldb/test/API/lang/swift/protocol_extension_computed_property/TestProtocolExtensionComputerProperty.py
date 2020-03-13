import lldbsuite.test.lldbinline as lldbinline
from lldbsuite.test.decorators import *

lldbinline.MakeInlineTest(
    __file__, globals(),
        decorators=[
            swiftTest,skipUnlessDarwin,
            skipIf(bugnumber="rdar://60396797", # should work but crashes.
                   setting=('symbols.use-swift-clangimporter', 'false'))
    ])
