from lldbsuite.test import lldbinline
from lldbsuite.test import decorators

lldbinline.MakeInlineTest(__file__, globals(),
        decorators.skipIf(archs=["armv7k", "i386"]))

