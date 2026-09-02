"""Test loading a corefile naming a binary that a symbol locator resolves to a
file which is not an object file."""

import os
import re
import subprocess

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCorefileUnparseableBinary(TestBase):
    @no_debug_info_test
    @requireDarwin
    @skipIfRemote
    def test(self):
        self.build()
        aout_exe = self.getBuildArtifact("a.out")
        corefile = self.getBuildArtifact("process.core")
        dsym_for_uuid = self.getBuildArtifact("dsym-for-uuid.sh")
        not_an_object_file = self.getBuildArtifact("not-an-object-file")
        hide_dir = self.getBuildArtifact("hide.noindex")
        lldbutil.mkdir_p(hide_dir)
        hide_aout_exe = self.getBuildArtifact("hide.noindex/a.out")

        dwarfdump_uuid_regex = re.compile(r"UUID: ([-0-9a-fA-F]+) \(([^\(]+)\) .*")
        dwarfdump_cmd_output = subprocess.check_output(
            ('/usr/bin/dwarfdump --uuid "%s"' % aout_exe), shell=True
        ).decode("utf-8")
        aout_uuid = None
        for line in dwarfdump_cmd_output.splitlines():
            match = dwarfdump_uuid_regex.search(line)
            if match:
                aout_uuid = match.group(1)
        self.assertNotEqual(aout_uuid, None, "Could not get uuid of built a.out")

        with open(not_an_object_file, "w") as writer:
            writer.write("this is not an object file\n")

        # Create our dsym-for-uuid shell script which returns a file that
        # exists but cannot be parsed as an object file.
        shell_cmds = [
            "#! /bin/sh",
            "# the last argument is the uuid",
            "while [ $# -gt 1 ]",
            "do",
            "  shift",
            "done",
            'echo "<?xml version=\\"1.0\\" encoding=\\"UTF-8\\"?>"',
            'echo "<!DOCTYPE plist PUBLIC \\"-//Apple//DTD PLIST 1.0//EN\\" \\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\\">"',
            'echo "<plist version=\\"1.0\\">"',
            'echo "<dict><key>$1</key><dict>"',
            "",
            'if [ "$1" = "%s" ]' % aout_uuid,
            "then",
            '  echo "<key>DBGSymbolRichExecutable</key><string>%s</string>"'
            % not_an_object_file,
            "else",
            '  echo "<key>DBGError</key><string>not found</string>"',
            "fi",
            "",
            'echo "</dict></dict></plist>"',
            "exit 0",
        ]

        with open(dsym_for_uuid, "w") as writer:
            for l in shell_cmds:
                writer.write(l + "\n")

        os.chmod(dsym_for_uuid, 0o755)

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c")
        )

        self.runCmd("process save-core --style=stack " + corefile)
        process.Kill()
        target.Clear()
        self.dbg.DeleteTarget(target)

        # Move a.out aside so the corefile's image can only be found by uuid,
        # and hook in the shell script that resolves that uuid.
        os.rename(aout_exe, hide_aout_exe)
        os.environ["LLDB_APPLE_DSYMFORUUID_EXECUTABLE"] = dsym_for_uuid
        self.addTearDownHook(
            lambda: os.environ.pop("LLDB_APPLE_DSYMFORUUID_EXECUTABLE", None)
        )

        target = self.dbg.CreateTarget("")
        process = target.LoadCore(corefile)
        self.assertTrue(process.IsValid())
        if self.TraceOn():
            self.runCmd("image list")

        # The image that could not be parsed has to be followed by another one,
        # or appending it to the Target is the last thing that happens.
        num_modules = target.GetNumModules()
        self.assertGreaterEqual(num_modules, 2)

        modules = [target.GetModuleAtIndex(i) for i in range(num_modules)]
        unparsed = [m for m in modules if m.GetNumSections() == 0]
        parsed = [m for m in modules if m.GetNumSections() > 0]

        self.assertEqual(len(unparsed), 1)
        self.assertGreaterEqual(len(parsed), 1)
