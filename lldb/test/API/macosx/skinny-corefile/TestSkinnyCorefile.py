"""Test that lldb can create a skinny corefile, and load all available libraries correctly."""



import os
import re
import subprocess

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestSkinnyCorefile(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfOutOfTreeDebugserver  # newer debugserver required for these qMemoryRegionInfo types
    @skipIf(debug_info=no_match(["dsym"]), bugnumber="This test is looking explicitly for a dSYM")
    @skipUnlessDarwin
    @skipIfRemote
    def test_lc_note(self):
        self.build()
        self.aout_exe = self.getBuildArtifact("a.out")
        self.aout_dsym = self.getBuildArtifact("a.out.dSYM")
        self.to_be_removed_dylib = self.getBuildArtifact("libto-be-removed.dylib")
        self.to_be_removed_dsym = self.getBuildArtifact("libto-be-removed.dylib.dSYM")
        self.corefile = self.getBuildArtifact("process.core")
        self.dsym_for_uuid = self.getBuildArtifact("dsym-for-uuid.sh")

        # After the corefile is created, we'll move a.out and a.out.dSYM 
        # into hide.noindex and lldb will have to use the 
        # LLDB_APPLE_DSYMFORUUID_EXECUTABLE script to find them.
        self.hide_dir = self.getBuildArtifact("hide.noindex")
        lldbutil.mkdir_p(self.hide_dir)
        self.hide_aout_exe = self.getBuildArtifact("hide.noindex/a.out")
        self.hide_aout_dsym = self.getBuildArtifact("hide.noindex/a.out.dSYM")

        # We can hook in our dsym-for-uuid shell script to lldb with 
        # this env var instead of requiring a defaults write.
        os.environ['LLDB_APPLE_DSYMFORUUID_EXECUTABLE'] = self.dsym_for_uuid
        self.addTearDownHook(lambda: os.environ.pop('LLDB_APPLE_DSYMFORUUID_EXECUTABLE', None))

        dwarfdump_uuid_regex = re.compile(
            'UUID: ([-0-9a-fA-F]+) \(([^\(]+)\) .*')
        dwarfdump_cmd_output = subprocess.check_output(
                ('/usr/bin/dwarfdump --uuid "%s"' % self.aout_exe), shell=True).decode("utf-8")
        aout_uuid = None
        for line in dwarfdump_cmd_output.splitlines():
            match = dwarfdump_uuid_regex.search(line)
            if match:
                aout_uuid = match.group(1)
        self.assertNotEqual(aout_uuid, None, "Could not get uuid of built a.out")

        ###  Create our dsym-for-uuid shell script which returns self.hide_aout_exe.
        shell_cmds = [
                '#! /bin/sh',
                '# the last argument is the uuid',
                'while [ $# -gt 1 ]',
                'do',
                '  shift',
                'done',
                'ret=0',
                'echo "<?xml version=\\"1.0\\" encoding=\\"UTF-8\\"?>"',
                'echo "<!DOCTYPE plist PUBLIC \\"-//Apple//DTD PLIST 1.0//EN\\" \\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\\">"',
                'echo "<plist version=\\"1.0\\">"',
                '',
                'if [ "$1" = "%s" ]' % aout_uuid,
                'then',
                '  uuid=%s' % aout_uuid,
                '  bin=%s' % self.hide_aout_exe,
                '  dsym=%s.dSYM/Contents/Resources/DWARF/%s' % (self.hide_aout_exe, os.path.basename(self.hide_aout_exe)),
                'fi',
                'if [ -z "$uuid" -o -z "$bin" -o ! -f "$bin" ]',
                'then',
                '  echo "<key>DBGError</key><string>not found</string>"',
                '  echo "</plist>"', 
                '  exit 1',
                'fi',
                'echo "<dict><key>$uuid</key><dict>"',
                '',
                'echo "<key>DBGArchitecture</key><string>x86_64</string>"',
                'echo "<key>DBGDSYMPath</key><string>$dsym</string>"',
                'echo "<key>DBGSymbolRichExecutable</key><string>$bin</string>"',
                'echo "</dict></dict></plist>"',
                'exit $ret'
                ]

        with open(self.dsym_for_uuid, "w") as writer:
            for l in shell_cmds:
                writer.write(l + '\n')

        os.chmod(self.dsym_for_uuid, 0o755)


        # Launch a live process with a.out, libto-be-removed.dylib, 
        # libpresent.dylib all in their original locations, create
        # a corefile at the breakpoint.
        (target, process, t, bp) = lldbutil.run_to_source_breakpoint (
                self, "break here", lldb.SBFileSpec('present.c'))

        self.assertTrue(process.IsValid())

        if self.TraceOn():
            self.runCmd("bt")
            self.runCmd("image list")

        self.runCmd("process save-core " + self.corefile)
        process.Kill()
        target.Clear()

        # Move the main binary and its dSYM into the hide.noindex
        # directory.  Now the only way lldb can find them is with
        # the LLDB_APPLE_DSYMFORUUID_EXECUTABLE shell script -
        # so we're testing that this dSYM discovery method works.
        os.rename(self.aout_exe, self.hide_aout_exe)
        os.rename(self.aout_dsym, self.hide_aout_dsym)

        # Completely remove the libto-be-removed.dylib, so we're
        # testing that lldb handles an unavailable binary correctly,
        # and non-dirty memory from this binary (e.g. the executing
        # instructions) are NOT included in the corefile.
        os.unlink(self.to_be_removed_dylib)
        shutil.rmtree(self.to_be_removed_dsym)


        # Now load the corefile
        self.target = self.dbg.CreateTarget('')
        self.process = self.target.LoadCore(self.corefile)
        self.assertTrue(self.process.IsValid())
        if self.TraceOn():
            self.runCmd("image list")
            self.runCmd("bt")

        self.assertTrue(self.process.IsValid())
        self.assertTrue(self.process.GetSelectedThread().IsValid())

        # f0 is present() in libpresent.dylib
        f0 = self.process.GetSelectedThread().GetFrameAtIndex(0)
        to_be_removed_dirty_data = f0.FindVariable("to_be_removed_dirty_data")
        self.assertEqual(to_be_removed_dirty_data.GetValueAsUnsigned(), 20)

        present_heap_buf = f0.FindVariable("present_heap_buf")
        self.assertIn("have ints 5 20 20 5", present_heap_buf.GetSummary())


        # f1 is to_be_removed() in libto-be-removed.dylib
        # it has been removed since the corefile was created,
        # and the instructions for this frame should NOT be included
        # in the corefile.  They were not dirty pages.
        f1 = self.process.GetSelectedThread().GetFrameAtIndex(1) 
        err = lldb.SBError()
        uint = self.process.ReadUnsignedFromMemory(f1.GetPC(), 4, err)
        self.assertTrue(err.Fail())


        # TODO Future testing could check that read-only constant data
        # (main_const_data, present_const_data) can be read both as an
        # SBValue and in an expression -- which means lldb needs to read
        # them out of the binaries, they are not present in the corefile.
        # And checking file-scope dirty data (main_dirty_data, 
        # present_dirty_data) the same way would be good, instead of just
        # checking the heap and stack like are being done right now.
