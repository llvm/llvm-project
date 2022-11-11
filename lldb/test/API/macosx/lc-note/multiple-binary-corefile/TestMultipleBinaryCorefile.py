"""Test corefiles with "main bin spec"/"load binary" with only addrs work."""


import os
import re
import subprocess

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestMultipleBinaryCorefile(TestBase):

    def initial_setup(self):
        self.build()
        self.aout_exe_basename = "a.out"
        self.libone_exe_basename = "libone.dylib"
        self.libtwo_exe_basename = "libtwo.dylib"
        self.aout_exe = self.getBuildArtifact(self.aout_exe_basename)
        self.aout_slide = 0x5000
        self.libone_exe = self.getBuildArtifact(self.libone_exe_basename)
        self.libone_slide = 0x100840000
        self.libtwo_exe = self.getBuildArtifact(self.libtwo_exe_basename)
        self.libtwo_slide = 0
        self.corefile = self.getBuildArtifact("multiple-binaries.core")
        self.create_corefile = self.getBuildArtifact("create-multibin-corefile")
        cmd="%s %s %s@%x %s@%x %s@%x" % (self.create_corefile, self.corefile, 
                                self.aout_exe, self.aout_slide,
                                self.libone_exe, self.libone_slide,
                                self.libtwo_exe, self.libtwo_slide)
        call(cmd, shell=True)


    def load_corefile_and_test(self):
        target = self.dbg.CreateTarget('')
        err = lldb.SBError()
        if self.TraceOn():
            self.runCmd("script print('loading corefile %s')" % self.corefile)
        process = target.LoadCore(self.corefile)
        self.assertEqual(process.IsValid(), True)
        if self.TraceOn():
            self.runCmd("script print('image list after loading corefile:')")
            self.runCmd("image list")

        self.assertEqual(target.GetNumModules(), 3)
        fspec = target.GetModuleAtIndex(0).GetFileSpec()
        self.assertEqual(fspec.GetFilename(), self.aout_exe_basename)

        # libone.dylib was never loaded into lldb, see that we added a memory module.
        fspec = target.GetModuleAtIndex(1).GetFileSpec()
        self.assertIn('memory-image', fspec.GetFilename())

        dwarfdump_uuid_regex = re.compile(
            'UUID: ([-0-9a-fA-F]+) \(([^\(]+)\) .*')
        dwarfdump_cmd_output = subprocess.check_output(
                ('/usr/bin/dwarfdump --uuid "%s"' % self.libone_exe), shell=True).decode("utf-8")
        libone_uuid = None
        for line in dwarfdump_cmd_output.splitlines():
            match = dwarfdump_uuid_regex.search(line)
            if match:
                libone_uuid = match.group(1)

        memory_image_uuid = target.GetModuleAtIndex(1).GetUUIDString()
        self.assertEqual(libone_uuid, memory_image_uuid)

        fspec = target.GetModuleAtIndex(2).GetFileSpec()
        self.assertEqual(fspec.GetFilename(), self.libtwo_exe_basename)

        # Executables "always" have this base address
        aout_load = target.GetModuleAtIndex(0).GetObjectFileHeaderAddress().GetLoadAddress(target)
        self.assertEqual(aout_load, 0x100000000 + self.aout_slide)

        # Value from Makefile
        libone_load = target.GetModuleAtIndex(1).GetObjectFileHeaderAddress().GetLoadAddress(target)
        self.assertEqual(libone_load, self.libone_slide)

        # Value from Makefile
        libtwo_load = target.GetModuleAtIndex(2).GetObjectFileHeaderAddress().GetLoadAddress(target)
        self.assertEqual(libtwo_load, self.libtwo_slide)

        self.dbg.DeleteTarget(target)
        self.dbg.Clear()

    NO_DEBUG_INFO_TESTCASE = True

    @skipIf(archs=no_match(['x86_64', 'arm64', 'arm64e', 'aarch64']))
    @skipIfRemote
    @skipUnlessDarwin
    def test_corefile_binaries_dsymforuuid(self):
        self.initial_setup()

        if self.TraceOn():
            self.runCmd("log enable lldb dyld host")
            self.addTearDownHook(lambda: self.runCmd("log disable lldb dyld host"))

        ## We can hook in our dsym-for-uuid shell script to lldb with this env
        ## var instead of requiring a defaults write.
        dsym_for_uuid = self.getBuildArtifact("dsym-for-uuid.sh")
        os.environ['LLDB_APPLE_DSYMFORUUID_EXECUTABLE'] = dsym_for_uuid
        if self.TraceOn():
            print("Setting env var LLDB_APPLE_DSYMFORUUID_EXECUTABLE=" + dsym_for_uuid)
        self.addTearDownHook(lambda: os.environ.pop('LLDB_APPLE_DSYMFORUUID_EXECUTABLE', None))

        self.runCmd("settings set target.load-script-from-symbol-file true")
        self.addTearDownHook(lambda: self.runCmd("settings set target.load-script-from-symbol-file false"))

        dwarfdump_uuid_regex = re.compile(
            'UUID: ([-0-9a-fA-F]+) \(([^\(]+)\) .*')
        dwarfdump_cmd_output = subprocess.check_output(
                ('/usr/bin/dwarfdump --uuid "%s"' % self.libtwo_exe), shell=True).decode("utf-8")
        libtwo_uuid = None
        for line in dwarfdump_cmd_output.splitlines():
            match = dwarfdump_uuid_regex.search(line)
            if match:
                libtwo_uuid = match.group(1)
        self.assertNotEqual(libtwo_uuid, None, "Could not get uuid of built libtwo.dylib")

        ###  Create our dsym-for-uuid shell script which returns aout_exe
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
                'if [ "$1" != "%s" ]' % (libtwo_uuid),
                'then',
                '  echo "<key>DBGError</key><string>not found</string>"',
                '  echo "</plist>"', 
                '  exit 1',
                'fi',
                '  uuid=%s' % libtwo_uuid,
                '  bin=%s' % self.libtwo_exe,
                '  dsym=%s.dSYM/Contents/Resources/DWARF/%s' % (self.libtwo_exe, os.path.basename(self.libtwo_exe)),
                'echo "<dict><key>$uuid</key><dict>"',
                '',
                'echo "<key>DBGDSYMPath</key><string>$dsym</string>"',
                'echo "<key>DBGSymbolRichExecutable</key><string>$bin</string>"',
                'echo "</dict></dict></plist>"',
                'exit $ret'
                ]

        with open(dsym_for_uuid, "w") as writer:
            for l in shell_cmds:
                writer.write(l + '\n')

        os.chmod(dsym_for_uuid, 0o755)

        # Register TWO of our binaries, but require dsymForUUID to find the third.
        target = self.dbg.CreateTarget(self.aout_exe, '', '', False, lldb.SBError())
        self.dbg.DeleteTarget(target)

        if self.TraceOn():
            self.runCmd("script print('Global image list, before loading corefile:')")
            self.runCmd("image list -g")

        self.load_corefile_and_test()

    @skipIf(archs=no_match(['x86_64', 'arm64', 'arm64e', 'aarch64']))
    @skipIfRemote
    @skipUnlessDarwin
    def test_corefile_binaries_preloaded(self):
        self.initial_setup()

        if self.TraceOn():
            self.runCmd("log enable lldb dyld host")
            self.addTearDownHook(lambda: self.runCmd("log disable lldb dyld host"))

        # Register all three binaries in lldb's global module
        # cache, then throw the Targets away.
        target = self.dbg.CreateTarget(self.aout_exe, '', '', False, lldb.SBError())
        self.dbg.DeleteTarget(target)
        target = self.dbg.CreateTarget(self.libtwo_exe, '', '', False, lldb.SBError())
        self.dbg.DeleteTarget(target)

        if self.TraceOn():
            self.runCmd("script print('Global image list, before loading corefile:')")
            self.runCmd("image list -g")

        self.load_corefile_and_test()
