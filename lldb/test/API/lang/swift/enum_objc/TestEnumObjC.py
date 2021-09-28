import lldbsuite.test.lldbinline as lldbinline
from lldbsuite.test.decorators import *

lldbinline.MakeInlineTest(
    __file__,
    globals(),
    decorators=[
        skipIf(
            bugnumber="rdar://60396797",  # should work but crashes.
            setting=('symbols.use-swift-clangimporter', 'false')),
        expectedFailureAll(oslist=["linux"], bugnumber="rdar://83444822"),
        swiftTest
    ])
