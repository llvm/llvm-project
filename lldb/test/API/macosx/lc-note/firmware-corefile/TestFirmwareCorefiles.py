"""Test that corefiles with LC_NOTE "kern ver str" and "main bin spec" load commands works."""



import os
import re
import subprocess

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestFirmwareCorefiles(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIf(debug_info=no_match(["dsym"]), bugnumber="This test is looking explicitly for a dSYM")
    @skipIf(archs=no_match(['x86_64']))
    @skipUnlessDarwin
    def test_lc_note(self):
        self.build()
        self.aout_exe = self.getBuildArtifact("a.out")
        self.bout_exe = self.getBuildArtifact("b.out")
        self.create_corefile = self.getBuildArtifact("create-empty-corefile")
        self.dsym_for_uuid = self.getBuildArtifact("dsym-for-uuid.sh")
        self.aout_corefile = self.getBuildArtifact("aout.core")
        self.bout_corefile = self.getBuildArtifact("bout.core")

        ## We can hook in our dsym-for-uuid shell script to lldb with this env
        ## var instead of requiring a defaults write.
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

        dwarfdump_cmd_output = subprocess.check_output(
                ('/usr/bin/dwarfdump --uuid "%s"' % self.bout_exe), shell=True).decode("utf-8")
        bout_uuid = None
        for line in dwarfdump_cmd_output.splitlines():
            match = dwarfdump_uuid_regex.search(line)
            if match:
                bout_uuid = match.group(1)
        self.assertNotEqual(bout_uuid, None, "Could not get uuid of built b.out")

        ###  Create our dsym-for-uuid shell script which returns self.aout_exe
        ###  or self.bout_exe, depending on the UUID on the command line.
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
                'if [ "$1" != "%s" -a "$1" != "%s" ]' % (aout_uuid, bout_uuid),
                'then',
                '  echo "<key>DBGError</key><string>not found</string>"',
                '  echo "</plist>"', 
                '  exit 1',
                'fi',
                'if [ "$1" = "%s" ]' % aout_uuid,
                'then',
                '  uuid=%s' % aout_uuid,
                '  bin=%s' % self.aout_exe,
                '  dsym=%s.dSYM/Contents/Resources/DWARF/%s' % (self.aout_exe, os.path.basename(self.aout_exe)),
                'else',
                '  uuid=%s' % bout_uuid,
                '  bin=%s' % self.bout_exe,
                '  dsym=%s.dSYM/Contents/Resources/DWARF/%s' % (self.bout_exe, os.path.basename(self.bout_exe)),
                'fi',
                'echo "<dict><key>$uuid</key><dict>"',
                '',
                'echo "<key>DBGArchitecture</key><string>i386</string>"',
                'echo "<key>DBGDSYMPath</key><string>$dsym</string>"',
                'echo "<key>DBGSymbolRichExecutable</key><string>$bin</string>"',
                'echo "</dict></dict></plist>"',
                'exit $ret'
                ]

        with open(self.dsym_for_uuid, "w") as writer:
            for l in shell_cmds:
                writer.write(l + '\n')

        os.chmod(self.dsym_for_uuid, 0o755)

        ### Create our corefile
        retcode = call(self.create_corefile + " version-string " + self.aout_corefile + " " + self.aout_exe, shell=True)
        retcode = call(self.create_corefile + " main-bin-spec " + self.bout_corefile + " " + self.bout_exe, shell=True)

        ### Now run lldb on the corefile
        ### which will give us a UUID
        ### which we call dsym-for-uuid.sh with
        ### which gives us a binary and dSYM
        ### which lldb should load!

        # First, try the "kern ver str" corefile
        self.target = self.dbg.CreateTarget('')
        err = lldb.SBError()
        self.process = self.target.LoadCore(self.aout_corefile)
        self.assertEqual(self.process.IsValid(), True)
        if self.TraceOn():
            self.runCmd("image list")
        self.assertEqual(self.target.GetNumModules(), 1)
        fspec = self.target.GetModuleAtIndex(0).GetFileSpec()
        filepath = fspec.GetDirectory() + "/" + fspec.GetFilename()
        self.assertEqual(filepath, self.aout_exe)


        # Second, try the "main bin spec" corefile
        self.target = self.dbg.CreateTarget('')
        self.process = self.target.LoadCore(self.bout_corefile)
        self.assertEqual(self.process.IsValid(), True)
        if self.TraceOn():
            self.runCmd("image list")
        self.assertEqual(self.target.GetNumModules(), 1)
        fspec = self.target.GetModuleAtIndex(0).GetFileSpec()
        filepath = fspec.GetDirectory() + "/" + fspec.GetFilename()
        self.assertEqual(filepath, self.bout_exe)
