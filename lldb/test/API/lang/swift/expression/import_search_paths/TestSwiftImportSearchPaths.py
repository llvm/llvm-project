import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import unittest2

class TestSwiftImportSearchPaths(lldbtest.TestBase):

    NO_DEBUG_INFO_TESTCASE = True

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

        log = self.getBuildArtifact("types.log")
        self.expect('settings show '
                    + 'target.experimental')
        self.expect("log enable lldb types -f " + log)
        self.expect("expr -- x.inner.hidden", substrs=['=', '42'])

        import io, re
        logfile = io.open(log, "r", encoding='utf-8')
        sanity = 0
        found = 0
        for line in logfile:
            if re.match(r'.*SwiftASTContextForModule\("a\.out"\)::LogConfiguration\(\).*hidden$',
                        line.strip('\n')):
                sanity += 1
            elif re.match(r'.*SwiftASTContextForExpressions::LogConfiguration\(\).*hidden$',
                          line.strip('\n')):
                found += 1
        self.assertEqual(sanity, 1)
        self.assertEqual(found, 1 if flag == 'true' else 0)
