import lldbsuite.test.lldbinline as lldbinline
from lldbsuite.test.decorators import *

lldbinline.MakeInlineTest(
    __file__,
    globals(),
    decorators=[
        expectedFailureAll(oslist=["linux"], bugnumber="rdar://83444822"),
        swiftTest
    ])
