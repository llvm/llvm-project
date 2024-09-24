import lldbsuite.test.lldbinline as lldbinline
from lldbsuite.test.decorators import *

# FIXME! The Swift driver insists on passing -experimental-skip-non-inlinable-function-bodies-without-types to -emit-module.
lldbinline.MakeInlineTest(__file__, globals(), decorators=[swiftTest])
