import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ObjCiVarIMPTestCase(TestBase):
    @skipUnlessDarwin
    @no_debug_info_test
    def test_imp_ivar_type(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_name_breakpoint(self, "main")
        frame = thread.GetFrameAtIndex(0)
        lang_info = frame.GetLanguageSpecificData()
        version = lang_info.GetValueForKey("Objective-C runtime version")
        self.assertEqual(version.GetIntegerValue(), 2)
