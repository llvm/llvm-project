import lldbsuite.test.lldbinline as lldbinline
from lldbsuite.test.decorators import *

lldbinline.MakeInlineTest(
    __file__,
    globals(),
    decorators=[
        requireNotEmbeddedSwift,
        skipIfLinux,  # https://github.com/swiftlang/llvm-project/issues/13465
        skipUnlessFoundationEssentials,
        swiftTest,
    ],
)
