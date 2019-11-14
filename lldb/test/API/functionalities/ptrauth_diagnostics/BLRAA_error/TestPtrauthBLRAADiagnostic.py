from lldbsuite.test import lldbinline
from lldbsuite.test import decorators

lldbinline.MakeInlineTest(__file__, globals(),
        [decorators.skipIf(archs=decorators.no_match(['arm64e']))])
