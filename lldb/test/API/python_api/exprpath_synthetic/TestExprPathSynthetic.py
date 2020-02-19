from lldbsuite.test import decorators
from lldbsuite.test import lldbinline

lldbinline.MakeInlineTest(
    __file__, globals(), [
        decorators.skipIfFreeBSD, decorators.skipIfLinux,
        decorators.skipIfWindows, decorators.skipIfNetBSD])
