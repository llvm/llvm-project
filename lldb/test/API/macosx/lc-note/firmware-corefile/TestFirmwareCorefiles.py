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

    def initial_setup(self):
        self.build()
        self.aout_exe = self.getBuildArtifact("a.out")
        self.aout_exe_basename = "a.out"
        self.create_corefile = self.getBuildArtifact("create-empty-corefile")
        self.dsym_for_uuid = self.getBuildArtifact("dsym-for-uuid.sh")
        self.verstr_corefile = self.getBuildArtifact("verstr.core")
        self.verstr_corefile_addr = self.getBuildArtifact("verstr-addr.core")
        self.binspec_corefile = self.getBuildArtifact("binspec.core")
        self.binspec_corefile_addr = self.getBuildArtifact("binspec-addr.core")

        ## We can hook in our dsym-for-uuid shell script to lldb with this env
        ## var instead of requiring a defaults write.
        os.environ['LLDB_APPLE_DSYMFORUUID_EXECUTABLE'] = self.dsym_for_uuid
        self.addTearDownHook(lambda: os.environ.pop('LLDB_APPLE_DSYMFORUUID_EXECUTABLE', None))

        self.runCmd("settings set target.load-script-from-symbol-file true")
        self.addTearDownHook(lambda: self.runCmd("settings set target.load-script-from-symbol-file false"))

        dsym_python_dir = '%s.dSYM/Contents/Resources/Python' % (self.aout_exe)
        os.makedirs(dsym_python_dir)
        python_os_plugin_path = os.path.join(self.getSourceDir(),
                                             'operating_system.py')
        python_init = [
                'def __lldb_init_module(debugger, internal_dict):',
                '  debugger.HandleCommand(\'settings set target.process.python-os-plugin-path %s\')' % python_os_plugin_path,
                ]
        with open(dsym_python_dir + "/a_out.py", "w") as writer:
            for l in python_init:
                writer.write(l + '\n')

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

        ###  Create our dsym-for-uuid shell script which returns self.aout_exe
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
                'if [ "$1" != "%s" ]' % (aout_uuid),
                'then',
                '  echo "<key>DBGError</key><string>not found</string>"',
                '  echo "</plist>"', 
                '  exit 1',
                'fi',
                '  uuid=%s' % aout_uuid,
                '  bin=%s' % self.aout_exe,
                '  dsym=%s.dSYM/Contents/Resources/DWARF/%s' % (self.aout_exe, os.path.basename(self.aout_exe)),
                'echo "<dict><key>$uuid</key><dict>"',
                '',
                'echo "<key>DBGDSYMPath</key><string>$dsym</string>"',
                'echo "<key>DBGSymbolRichExecutable</key><string>$bin</string>"',
                'echo "</dict></dict></plist>"',
                'exit $ret'
                ]

        with open(self.dsym_for_uuid, "w") as writer:
            for l in shell_cmds:
                writer.write(l + '\n')

        os.chmod(self.dsym_for_uuid, 0o755)

        self.slide = 0x70000000000

        ### Create our corefile
        # 0xffffffffffffffff means load address unknown
        retcode = call(self.create_corefile + " version-string " + self.verstr_corefile + " " + self.aout_exe + " 0xffffffffffffffff", shell=True)
        retcode = call(self.create_corefile + " version-string " + self.verstr_corefile_addr + " " + self.aout_exe + (" 0x%x" % self.slide), shell=True)
        retcode = call(self.create_corefile + " main-bin-spec " + self.binspec_corefile + " " + self.aout_exe + " 0xffffffffffffffff", shell=True)
        retcode = call(self.create_corefile + " main-bin-spec " + self.binspec_corefile_addr + " " + self.aout_exe + (" 0x%x" % self.slide), shell=True)

    @skipIf(debug_info=no_match(["dsym"]), bugnumber="This test is looking explicitly for a dSYM")
    @skipIf(archs=no_match(['x86_64']))
    @skipUnlessDarwin
    def test_lc_note_version_string(self):
        self.initial_setup()

        if self.TraceOn():
            self.runCmd("log enable lldb dyld host")
            self.addTearDownHook(lambda: self.runCmd("log disable lldb dyld host"))

        ### Now run lldb on the corefile
        ### which will give us a UUID
        ### which we call dsym-for-uuid.sh with
        ### which gives us a binary and dSYM
        ### which lldb should load!

        # First, try the "kern ver str" corefile
        self.target = self.dbg.CreateTarget('')
        err = lldb.SBError()
        if self.TraceOn():
            self.runCmd("script print('loading corefile %s')" % self.verstr_corefile)
        self.process = self.target.LoadCore(self.verstr_corefile)
        self.assertEqual(self.process.IsValid(), True)
        if self.TraceOn():
            self.runCmd("image list")
            self.runCmd("target mod dump sections")
        self.assertEqual(self.target.GetNumModules(), 1)
        fspec = self.target.GetModuleAtIndex(0).GetFileSpec()
        self.assertEqual(fspec.GetFilename(), self.aout_exe_basename)
        self.process.Kill()
        self.process = None
        self.target.Clear()
        self.target = None
        self.dbg.MemoryPressureDetected()

        # Second, try the "kern ver str" corefile where it loads at an address
        self.target = self.dbg.CreateTarget('')
        err = lldb.SBError()
        if self.TraceOn():
            self.runCmd("script print('loading corefile %s')" % self.verstr_corefile_addr)
        self.process = self.target.LoadCore(self.verstr_corefile_addr)
        self.assertEqual(self.process.IsValid(), True)
        if self.TraceOn():
            self.runCmd("image list")
            self.runCmd("target mod dump sections")
        self.assertEqual(self.target.GetNumModules(), 1)
        fspec = self.target.GetModuleAtIndex(0).GetFileSpec()
        self.assertEqual(fspec.GetFilename(), self.aout_exe_basename)
        main_sym = self.target.GetModuleAtIndex(0).FindSymbol("main", lldb.eSymbolTypeAny)
        main_addr = main_sym.GetStartAddress()
        self.assertGreater(main_addr.GetLoadAddress(self.target), self.slide)
        self.assertNotEqual(main_addr.GetLoadAddress(self.target), lldb.LLDB_INVALID_ADDRESS)
        self.process.Kill()
        self.process = None
        self.target.Clear()
        self.target = None
        self.dbg.MemoryPressureDetected()

    @skipIf(debug_info=no_match(["dsym"]), bugnumber="This test is looking explicitly for a dSYM")
    @skipIf(archs=no_match(['x86_64']))
    @skipUnlessDarwin
    def test_lc_note_main_bin_spec(self):
        self.initial_setup()

        if self.TraceOn():
            self.runCmd("log enable lldb dyld host")
            self.addTearDownHook(lambda: self.runCmd("log disable lldb dyld host"))

        # Third, try the "main bin spec" corefile
        self.target = self.dbg.CreateTarget('')
        if self.TraceOn():
            self.runCmd("script print('loading corefile %s')" % self.binspec_corefile)
        self.process = self.target.LoadCore(self.binspec_corefile)
        self.assertEqual(self.process.IsValid(), True)
        if self.TraceOn():
            self.runCmd("image list")
            self.runCmd("target mod dump sections")
        self.assertEqual(self.target.GetNumModules(), 1)
        fspec = self.target.GetModuleAtIndex(0).GetFileSpec()
        self.assertEqual(fspec.GetFilename(), self.aout_exe_basename)
        self.process.Kill()
        self.process = None
        self.target.Clear()
        self.target = None
        self.dbg.MemoryPressureDetected()

        # Fourth, try the "main bin spec" corefile where it loads at an address
        self.target = self.dbg.CreateTarget('')
        if self.TraceOn():
            self.runCmd("script print('loading corefile %s')" % self.binspec_corefile_addr)
        self.process = self.target.LoadCore(self.binspec_corefile_addr)
        self.assertEqual(self.process.IsValid(), True)
        if self.TraceOn():
            self.runCmd("image list")
            self.runCmd("target mod dump sections")
        self.assertEqual(self.target.GetNumModules(), 1)
        fspec = self.target.GetModuleAtIndex(0).GetFileSpec()
        self.assertEqual(fspec.GetFilename(), self.aout_exe_basename)
        main_sym = self.target.GetModuleAtIndex(0).FindSymbol("main", lldb.eSymbolTypeAny)
        main_addr = main_sym.GetStartAddress()
        self.assertGreater(main_addr.GetLoadAddress(self.target), self.slide)
        self.assertNotEqual(main_addr.GetLoadAddress(self.target), lldb.LLDB_INVALID_ADDRESS)
        self.process.Kill()
        self.process = None
        self.target.Clear()
        self.target = None
        self.dbg.MemoryPressureDetected()

    @skipIf(debug_info=no_match(["dsym"]), bugnumber="This test is looking explicitly for a dSYM")
    @skipIf(archs=no_match(['x86_64']))
    @skipUnlessDarwin
    def test_lc_note_main_bin_spec_os_plugin(self):
        self.initial_setup()

        if self.TraceOn():
            self.runCmd("log enable lldb dyld host")
            self.addTearDownHook(lambda: self.runCmd("log disable lldb dyld host"))
        # Now load the binary and confirm that we load the OS plugin.
        self.target = self.dbg.CreateTarget('')

        if self.TraceOn():
            self.runCmd("script print('loading corefile %s with OS plugin')" % self.binspec_corefile_addr)
        self.process = self.target.LoadCore(self.binspec_corefile_addr)
        self.assertEqual(self.process.IsValid(), True)
        if self.TraceOn():
            self.runCmd("image list")
            self.runCmd("target mod dump sections")
            self.runCmd("thread list")
        self.assertEqual(self.target.GetNumModules(), 1)
        fspec = self.target.GetModuleAtIndex(0).GetFileSpec()
        self.assertEqual(fspec.GetFilename(), self.aout_exe_basename)

        # Verify our OS plug-in threads showed up
        thread = self.process.GetThreadByID(0x111111111)
        self.assertTrue(thread.IsValid(), 
                "Make sure there is a thread 0x111111111 after we load the python OS plug-in")
        thread = self.process.GetThreadByID(0x222222222)
        self.assertTrue(thread.IsValid(), 
                "Make sure there is a thread 0x222222222 after we load the python OS plug-in")
        thread = self.process.GetThreadByID(0x333333333)
        self.assertTrue(thread.IsValid(), 
                "Make sure there is a thread 0x333333333 after we load the python OS plug-in")

        self.process.Kill()
        self.process = None
        self.target.Clear()
        self.target = None
        self.dbg.MemoryPressureDetected()
