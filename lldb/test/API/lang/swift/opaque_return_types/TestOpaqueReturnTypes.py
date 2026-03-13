import lldbsuite.test.lldbinline as lldbinline
from lldbsuite.test.decorators import *

lldbinline.MakeInlineTest(
    __file__,
    globals(),
    decorators=[
        swiftTest,
        expectedFailureWindows,
        skipIf(macos_version=["<", "10.15"]),
    ],
)
