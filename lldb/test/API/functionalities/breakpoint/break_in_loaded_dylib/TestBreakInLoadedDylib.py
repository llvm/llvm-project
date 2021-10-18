import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestBreakInLoadedDylib(TestBase):
    """ Test that we can set a source regex breakpoint that will take in
    a dlopened library that hasn't loaded when we set the breakpoint."""

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfRemote
    def common_setup(self):
        self.build()
        ctx = self.platformContext
        self.main_spec = lldb.SBFileSpec("main.cpp")
        self.b_spec = lldb.SBFileSpec("b.cpp")
        self.lib_shortname = 'lib_b'
        self.lib_fullname = ctx.shlib_prefix + self.lib_shortname + '.' + ctx.shlib_extension
        self.lib_spec = lldb.SBFileSpec(self.lib_fullname)
        
    def test_break_in_dlopen_dylib_using_lldbutils(self):
        self.common_setup()
        lldbutil.run_to_source_breakpoint(self, "Break here in dylib", self.b_spec,
                                          bkpt_module=self.lib_fullname,
                                          extra_images = [self.lib_shortname],
                                          has_locations_before_run = False)

    @skipIfRemote
    def test_break_in_dlopen_dylib_using_target(self):
        self.common_setup()

        target, process, _, _ = lldbutil.run_to_source_breakpoint(self, "Break here before we dlopen", self.main_spec,
                                                            extra_images = [self.lib_shortname])
        
        # Now set some breakpoints that won't take till the library is loaded:
        # This one is currently how lldbutils does it but test here in case that changes:
        bkpt1 = target.BreakpointCreateBySourceRegex("Break here in dylib", self.b_spec, self.lib_fullname)
        self.assertEqual(bkpt1.GetNumLocations(), 0, "Library isn't loaded yet.")
        # Try the file list API as well.  Put in some bogus entries too, to make sure those
        # don't trip us up:
                                               
        files_list = lldb.SBFileSpecList()
        files_list.Append(self.b_spec)
        files_list.Append(self.main_spec)
        files_list.Append(lldb.SBFileSpec("I_bet_nobody_has_this_file.cpp"))

        modules_list = lldb.SBFileSpecList()
        modules_list.Append(self.lib_spec)
        modules_list.Append(lldb.SBFileSpec("libI_bet_not_this_one_either.dylib"))

        bkpt2 = target.BreakpointCreateBySourceRegex("Break here in dylib", modules_list, files_list)
        self.assertEqual(bkpt2.GetNumLocations(), 0, "Library isn't loaded yet")

        lldbutil.continue_to_breakpoint(process, bkpt1)
        self.assertEqual(bkpt1.GetHitCount(), 1, "Hit breakpoint 1")
        self.assertEqual(bkpt2.GetHitCount(), 1, "Hit breakpoint 2")

        

        
