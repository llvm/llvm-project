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
    @skipIf(archs=no_match(['x86_64', 'arm64', 'arm64e', 'aarch64']))
    @skipIfRemote
    @skipUnlessDarwin
    def test_lc_note_version_string(self):
        self.build()
        aout_exe_basename = "a.out"
        aout_exe = self.getBuildArtifact(aout_exe_basename)
        verstr_corefile = self.getBuildArtifact("verstr.core")
        verstr_corefile_addr = self.getBuildArtifact("verstr-addr.core")
        create_corefile = self.getBuildArtifact("create-empty-corefile")
        slide = 0x70000000000
        call(create_corefile + " version-string " + verstr_corefile + " " + aout_exe + " 0xffffffffffffffff 0xffffffffffffffff", shell=True)
        call(create_corefile + " version-string " + verstr_corefile_addr + " " + aout_exe + (" 0x%x" % slide) + " 0xffffffffffffffff", shell=True)

        if self.TraceOn():
            self.runCmd("log enable lldb dyld host")
            self.addTearDownHook(lambda: self.runCmd("log disable lldb dyld host"))

        # Register the a.out binary with this UUID in lldb's global module
        # cache, then throw the Target away.
        target = self.dbg.CreateTarget(aout_exe)
        self.dbg.DeleteTarget(target)

        # First, try the "kern ver str" corefile
        target = self.dbg.CreateTarget('')
        err = lldb.SBError()
        if self.TraceOn():
            self.runCmd("script print('loading corefile %s')" % verstr_corefile)
        process = target.LoadCore(verstr_corefile)
        self.assertEqual(process.IsValid(), True)
        if self.TraceOn():
            self.runCmd("image list")
            self.runCmd("target mod dump sections")
        self.assertEqual(target.GetNumModules(), 1)
        fspec = target.GetModuleAtIndex(0).GetFileSpec()
        self.assertEqual(fspec.GetFilename(), aout_exe_basename)
        self.dbg.DeleteTarget(target)

        # Second, try the "kern ver str" corefile where it loads at an address
        target = self.dbg.CreateTarget('')
        err = lldb.SBError()
        if self.TraceOn():
            self.runCmd("script print('loading corefile %s')" % verstr_corefile_addr)
        process = target.LoadCore(verstr_corefile_addr)
        self.assertEqual(process.IsValid(), True)
        if self.TraceOn():
            self.runCmd("image list")
            self.runCmd("target mod dump sections")
        self.assertEqual(target.GetNumModules(), 1)
        fspec = target.GetModuleAtIndex(0).GetFileSpec()
        self.assertEqual(fspec.GetFilename(), aout_exe_basename)
        main_sym = target.GetModuleAtIndex(0).FindSymbol("main", lldb.eSymbolTypeAny)
        main_addr = main_sym.GetStartAddress()
        self.assertGreater(main_addr.GetLoadAddress(target), slide)
        self.assertNotEqual(main_addr.GetLoadAddress(target), lldb.LLDB_INVALID_ADDRESS)
        self.dbg.DeleteTarget(target)

    @skipIf(debug_info=no_match(["dsym"]), bugnumber="This test is looking explicitly for a dSYM")
    @skipIf(archs=no_match(['x86_64', 'arm64', 'arm64e', 'aarch64']))
    @skipIfRemote
    @skipUnlessDarwin
    def test_lc_note_main_bin_spec(self):
        self.build()
        aout_exe_basename = "a.out"
        aout_exe = self.getBuildArtifact(aout_exe_basename)
        create_corefile = self.getBuildArtifact("create-empty-corefile")
        binspec_corefile = self.getBuildArtifact("binspec.core")
        binspec_corefile_addr = self.getBuildArtifact("binspec-addr.core")
        binspec_corefile_slideonly = self.getBuildArtifact("binspec-addr-slideonly.core")

        slide = 0x70000000000

        ### Create our corefile
        # 0xffffffffffffffff means load address unknown
        call(create_corefile + " main-bin-spec " + binspec_corefile + " " + aout_exe + " 0xffffffffffffffff 0xffffffffffffffff", shell=True)
        call(create_corefile + " main-bin-spec " + binspec_corefile_addr + " " + aout_exe + (" 0x%x" % slide) + " 0xffffffffffffffff", shell=True)
        call(create_corefile + " main-bin-spec " + binspec_corefile_slideonly + " " + aout_exe + " 0xffffffffffffffff" + (" 0x%x" % slide), shell=True)

        if self.TraceOn():
            self.runCmd("log enable lldb dyld host")
            self.addTearDownHook(lambda: self.runCmd("log disable lldb dyld host"))

        # Register the a.out binary with this UUID in lldb's global module
        # cache, then throw the Target away.
        target = self.dbg.CreateTarget(aout_exe)
        self.dbg.DeleteTarget(target)

        # First, try the "main bin spec" corefile
        target = self.dbg.CreateTarget('')
        if self.TraceOn():
            self.runCmd("script print('loading corefile %s')" % binspec_corefile)
        process = target.LoadCore(binspec_corefile)
        self.assertEqual(process.IsValid(), True)
        if self.TraceOn():
            self.runCmd("image list")
            self.runCmd("target mod dump sections")
        self.assertEqual(target.GetNumModules(), 1)
        fspec = target.GetModuleAtIndex(0).GetFileSpec()
        self.assertEqual(fspec.GetFilename(), aout_exe_basename)
        self.dbg.DeleteTarget(target)

        # Second, try the "main bin spec" corefile where it loads at an address
        target = self.dbg.CreateTarget('')
        if self.TraceOn():
            self.runCmd("script print('loading corefile %s')" % binspec_corefile_addr)
        process = target.LoadCore(binspec_corefile_addr)
        self.assertEqual(process.IsValid(), True)
        if self.TraceOn():
            self.runCmd("image list")
            self.runCmd("target mod dump sections")
        self.assertEqual(target.GetNumModules(), 1)
        fspec = target.GetModuleAtIndex(0).GetFileSpec()
        self.assertEqual(fspec.GetFilename(), aout_exe_basename)
        main_sym = target.GetModuleAtIndex(0).FindSymbol("main", lldb.eSymbolTypeAny)
        main_addr = main_sym.GetStartAddress()
        self.assertGreater(main_addr.GetLoadAddress(target), slide)
        self.assertNotEqual(main_addr.GetLoadAddress(target), lldb.LLDB_INVALID_ADDRESS)
        self.dbg.DeleteTarget(target)

        # Third, try the "main bin spec" corefile where it loads at a slide
        target = self.dbg.CreateTarget('')
        if self.TraceOn():
            self.runCmd("script print('loading corefile %s')" % binspec_corefile_slideonly)
        process = target.LoadCore(binspec_corefile_slideonly)
        self.assertEqual(process.IsValid(), True)
        if self.TraceOn():
            self.runCmd("image list")
            self.runCmd("target mod dump sections")
        self.assertEqual(target.GetNumModules(), 1)
        fspec = target.GetModuleAtIndex(0).GetFileSpec()
        self.assertEqual(fspec.GetFilename(), aout_exe_basename)
        main_sym = target.GetModuleAtIndex(0).FindSymbol("main", lldb.eSymbolTypeAny)
        main_addr = main_sym.GetStartAddress()
        self.assertGreater(main_addr.GetLoadAddress(target), slide)
        self.assertNotEqual(main_addr.GetLoadAddress(target), lldb.LLDB_INVALID_ADDRESS)
        self.dbg.DeleteTarget(target)

    @skipIf(debug_info=no_match(["dsym"]), bugnumber="This test is looking explicitly for a dSYM")
    @skipIf(archs=no_match(['x86_64', 'arm64', 'arm64e', 'aarch64']))
    @skipIfRemote
    @skipUnlessDarwin
    def test_lc_note_main_bin_spec_os_plugin(self):

        self.build()
        aout_exe = self.getBuildArtifact("a.out")
        aout_exe_basename = "a.out"
        create_corefile = self.getBuildArtifact("create-empty-corefile")
        binspec_corefile_addr = self.getBuildArtifact("binspec-addr.core")

        slide = 0x70000000000

        ### Create our corefile
        # 0xffffffffffffffff means load address unknown
        call(create_corefile + " main-bin-spec " + binspec_corefile_addr + " " + aout_exe + (" 0x%x" % slide) + " 0xffffffffffffffff", shell=True)

        ## We can hook in our dsym-for-uuid shell script to lldb with this env
        ## var instead of requiring a defaults write.
        dsym_for_uuid = self.getBuildArtifact("dsym-for-uuid.sh")
        os.environ['LLDB_APPLE_DSYMFORUUID_EXECUTABLE'] = dsym_for_uuid
        if self.TraceOn():
            print("Setting env var LLDB_APPLE_DSYMFORUUID_EXECUTABLE=" + dsym_for_uuid)
        self.addTearDownHook(lambda: os.environ.pop('LLDB_APPLE_DSYMFORUUID_EXECUTABLE', None))

        self.runCmd("settings set target.load-script-from-symbol-file true")
        self.addTearDownHook(lambda: self.runCmd("settings set target.load-script-from-symbol-file false"))

        dsym_python_dir = os.path.join('%s.dSYM' % aout_exe, 'Contents', 'Resources', 'Python')
        os.makedirs(dsym_python_dir)
        python_os_plugin_path = os.path.join(self.getSourceDir(),
                                             'operating_system.py')
        python_init = [
                'def __lldb_init_module(debugger, internal_dict):',
                '  debugger.HandleCommand(\'settings set target.process.python-os-plugin-path %s\')' % python_os_plugin_path,
                ]
        with open(os.path.join(dsym_python_dir, "a_out.py"), "w") as writer:
            for l in python_init:
                writer.write(l + '\n')

        dwarfdump_uuid_regex = re.compile(
            'UUID: ([-0-9a-fA-F]+) \(([^\(]+)\) .*')
        dwarfdump_cmd_output = subprocess.check_output(
                ('/usr/bin/dwarfdump --uuid "%s"' % aout_exe), shell=True).decode("utf-8")
        aout_uuid = None
        for line in dwarfdump_cmd_output.splitlines():
            match = dwarfdump_uuid_regex.search(line)
            if match:
                aout_uuid = match.group(1)
        self.assertNotEqual(aout_uuid, None, "Could not get uuid of built a.out")

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
                'if [ "$1" != "%s" ]' % (aout_uuid),
                'then',
                '  echo "<key>DBGError</key><string>not found</string>"',
                '  echo "</plist>"', 
                '  exit 1',
                'fi',
                '  uuid=%s' % aout_uuid,
                '  bin=%s' % aout_exe,
                '  dsym=%s.dSYM/Contents/Resources/DWARF/%s' % (aout_exe, os.path.basename(aout_exe)),
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

        ### Now run lldb on the corefile
        ### which will give us a UUID
        ### which we call dsym-for-uuid.sh with
        ### which gives us a binary and dSYM
        ### which lldb should load!

        if self.TraceOn():
            self.runCmd("log enable lldb dyld host")
            self.addTearDownHook(lambda: self.runCmd("log disable lldb dyld host"))
        # Now load the binary and confirm that we load the OS plugin.
        target = self.dbg.CreateTarget('')

        if self.TraceOn():
            self.runCmd("script print('loading corefile %s with OS plugin')" % binspec_corefile_addr)

        process = target.LoadCore(binspec_corefile_addr)
        self.assertEqual(process.IsValid(), True)
        if self.TraceOn():
            self.runCmd("image list")
            self.runCmd("target mod dump sections")
            self.runCmd("thread list")
        self.assertEqual(target.GetNumModules(), 1)
        fspec = target.GetModuleAtIndex(0).GetFileSpec()
        self.assertEqual(fspec.GetFilename(), aout_exe_basename)

        # Verify our OS plug-in threads showed up
        thread = process.GetThreadByID(0x111111111)
        self.assertTrue(thread.IsValid(), 
                "Make sure there is a thread 0x111111111 after we load the python OS plug-in")
        thread = process.GetThreadByID(0x222222222)
        self.assertTrue(thread.IsValid(), 
                "Make sure there is a thread 0x222222222 after we load the python OS plug-in")
        thread = process.GetThreadByID(0x333333333)
        self.assertTrue(thread.IsValid(), 
                "Make sure there is a thread 0x333333333 after we load the python OS plug-in")

        self.runCmd("settings clear target.process.python-os-plugin-path")
        self.dbg.DeleteTarget(target)
