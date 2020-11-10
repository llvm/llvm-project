from lldbsuite.test import decorators
from lldbsuite.test import lldbinline

lldbinline.MakeInlineTest(
    __file__, globals(), [
        decorators.skipUnlessDarwin,
        decorators.expectedFailureAll(
            oslist=['macosx'], archs=['i386'],
            bugnumber='rdar://28656677')])
