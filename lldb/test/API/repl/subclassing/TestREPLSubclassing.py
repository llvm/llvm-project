import lldbsuite.test.lldbinrepl as lldbinrepl
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.decorators as decorators

lldbinrepl.MakeREPLTest(__file__, globals(),
        decorators=[decorators.skipIfDarwin, decorators.skipUnlessDarwin])
