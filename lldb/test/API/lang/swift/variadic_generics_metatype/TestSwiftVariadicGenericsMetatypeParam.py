import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class SwiftVariadicGenericsMetatypeParamTest(TestBase):
    @skipEmbeddedSwiftOnWindows # Embedded Swift doesn't link on Windows (posix_memalign).
    @swiftTest
    def test(self):
        """A generic value carrying `repeat each T` resolves under a function whose parameter is `repeat (each T).Type`."""
        self.build()
        self.expect('settings set symbols.swift-enable-ast-context false')
        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        # FIXME: rewrite this using check_variable. Unfortunately the
        # substituted type isn't correctly reported in the SBAPI.
        self.expect('v x',
                    substrs=['(a.OuterStruct<Pack{Int, String}>) x'])

        # Same-shape sibling case: in `entrySameShape<each T, each U>` the
        # function's only pack expansion has shape T, but the variable's
        # pattern is `each U`.
        process.Continue()
        self.expect('v y',
                    substrs=['(a.OuterStruct<Pack{String, String}>) y'])
