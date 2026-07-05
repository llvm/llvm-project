from lldbsuite.test import decorators
from lldbsuite.test import lldbinline

lldbinline.MakeInlineTest(
    __file__,
    globals(),
    [
        decorators.add_test_categories(["objc"]),
        decorators.expectedFailureAll(
            oslist=["macosx"], archs=["i386"], bugnumber="rdar://28656677"
        ),
    ],
)
