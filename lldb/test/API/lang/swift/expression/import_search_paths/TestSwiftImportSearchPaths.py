import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import unittest2

class TestSwiftImportSearchPaths(lldbtest.TestBase):

    NO_DEBUG_INFO_TESTCASE = True
    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @swiftTest
    def test_positive(self):
        self.do_test('true')

    @swiftTest
    def test_negative(self):
        self.do_test('false')
        
    def do_test(self, flag):
        """Test a .swiftmodule that was compiled with serialized debugging
           options, using a search path to another module it imports. We then
           need to build a (third) Swift (application) module with search paths
           to #1 but not to #2, relying on the serialized options."""
        self.build()
        self.expect('settings set '
                    + 'target.experimental.swift-discover-implicit-search-paths '
                    + flag)
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'),
            extra_images=['Direct', self.getBuildArtifact('hidden/libIndirect')])

        types_log = self.getBuildArtifact("types.log")
        self.expect("log enable lldb types -f " + types_log)
        self.expect("expr -- x.inner.hidden", substrs=['=', '42'])
        if flag == 'true':
            prefix = 'POSITIVE'
        else:
            prefix = 'NEGATIVE'
        self.filecheck('platform shell cat "%s"' % types_log, __file__,
                       '--check-prefix=CHECK_MOD_'+prefix)
        self.filecheck('platform shell cat "%s"' % types_log, __file__,
                       '--check-prefix=CHECK_EXP_'+prefix)
# CHECK_MOD_POSITIVE: SwiftASTContextForModule("a.out")::LogConfiguration(){{.*hidden$}}
# CHECK_MOD_NEGATIVE: SwiftASTContextForModule("a.out")::LogConfiguration(){{.*hidden$}}
# CHECK_EXP_POSITIVE: SwiftASTContextForExpressions::LogConfiguration(){{.*hidden$}}
# CHECK_EXP_NEGATIVE-NOT: SwiftASTContextForExpressions::LogConfiguration(){{.*hidden$}}
# CHECK_EXP_NEGATIVE: SwiftASTContextForExpressions::LogConfiguration(){{.*}}Extra clang arguments
