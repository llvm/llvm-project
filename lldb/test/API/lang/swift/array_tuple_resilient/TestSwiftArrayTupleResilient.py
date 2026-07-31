import lldbsuite.test.lldbinline as lldbinline
from lldbsuite.test.decorators import *

lldbinline.MakeInlineTest(__file__, globals(), decorators=[skipEmbeddedSwift,
        swiftTest, skipIfLinux,
    expectedFailureAll(archs=['arm64_32'], bugnumber="<rdar://problem/58065423>")])
