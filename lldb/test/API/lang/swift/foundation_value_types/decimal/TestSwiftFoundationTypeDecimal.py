import lldbsuite.test.lldbinline as lldbinline
from lldbsuite.test.decorators import *

lldbinline.MakeInlineTest(__file__, globals(),
                          decorators=[swiftTest,skipIf(oslist=['windows']),
expectedFailureAll(oslist=['macosx'],bugnumber="rdar://60396797",
                   setting=('symbols.use-swift-clangimporter', 'false'))
])
