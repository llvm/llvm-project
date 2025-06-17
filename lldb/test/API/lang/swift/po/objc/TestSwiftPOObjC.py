import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftPOObjC(TestBase):
    #NO_DEBUG_INFO_TESTCASE = True
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        """Test running po on a Swift object from Objective-C. This
        should initialize a generic SwiftASTContext with no parsing of
        Swift modules happening"""
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec('main.m'))

        log = self.getBuildArtifact("types.log")
        self.expect('log enable lldb types -f "%s"' % log)
        self.expect("expr -O -- base", substrs=["Hello from Swift"])
        self.filecheck('platform shell cat "%s"' % log, __file__)
### -cc1 should be round-tripped so there is no more `-cc1` in the extra args. Look for `-triple` which is a cc1 flag.
#       CHECK-NOT: parsed module "a"
#       CHECK:  SwiftASTContextForExpressions(module: "{{.*-.*-.*}}", cu: "*")::LogConfiguration()
        self.expect("expr -- base", substrs=["a.Bar"])
        self.expect("expr -- *base", substrs=["a.Bar"])
