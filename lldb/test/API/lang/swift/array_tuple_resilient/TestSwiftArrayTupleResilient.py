import lldbsuite.test.lldbinline as lldbinline
from lldbsuite.test.decorators import *

lldbinline.MakeInlineTest(__file__, globals(), decorators=[swiftTest, skipUnlessDarwin,
expectedFailureAll(bugnumber="rdar://60396797",
                        setting=('symbols.use-swift-clangimporter', 'false'))
])
