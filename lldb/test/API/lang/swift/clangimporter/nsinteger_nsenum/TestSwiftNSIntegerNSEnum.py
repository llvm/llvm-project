import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftNSIntegerNSEnum(lldbtest.TestBase):
    def do_test(self, use_summary):
        def check(var, output):
            if use_summary:
                lldbutil.check_variable(self, var, False, summary='.'+output)
            else:
                lldbutil.check_variable(self, var, False, value=output)
        
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        frame = thread.frames[0]
        self.assertTrue(frame)
        check(frame.FindVariable('e1'), 'eCase1')
        check(frame.FindVariable('e2'), 'eCase2')

    @skipUnlessDarwin
    @swiftTest
    def test_reflection(self):
        self.expect("setting set symbols.swift-enable-ast-context false")
        self.do_test(use_summary=True)

    # Don't run a clangimporter test without ClangImporter.
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @skipUnlessDarwin
    @swiftTest
    def test_swift_ast(self):
        self.expect("setting set symbols.swift-enable-ast-context true")
        self.do_test(use_summary=False)
