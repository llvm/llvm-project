import lldbsuite.test.lldbinrepl as lldbinrepl
import lldbsuite.test.lldbtest as lldbtest
from lldbsuite.test.decorators import *

lldbinrepl.MakeREPLTest(__file__, globals(), decorators=[swiftTest])
