import lldbsuite.test.lldbinline as lldbinline
from lldbsuite.test.decorators import *

lldbinline.MakeInlineTest(
    __file__, globals(),
        decorators=[
            expectedFailureAll, #FIXME: This regressed silently due to 2c911bceb06ed376801251bdfd992905a66f276c
            swiftTest,skipUnlessDarwin,
            skipIf(bugnumber="rdar://60396797", # should work but crashes.
                   setting=('symbols.use-swift-clangimporter', 'false'))
    ])
