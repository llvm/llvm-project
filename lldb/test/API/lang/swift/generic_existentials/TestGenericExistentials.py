import lldbsuite.test.lldbinline as lldbinline
from lldbsuite.test.decorators import *

lldbinline.MakeInlineTest(__file__, globals(), decorators=[swiftTest,
                          expectedFailureAll #FIXME: This regressed silently due to 2c911bceb06ed376801251bdfd992905a66f276c
                          ])
