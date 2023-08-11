"""
An SBTarget with no arch, call AddModule, SBTarget's arch should be set.
"""

import os
import subprocess
import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TargetArchFromModule(TestBase):
    @skipIf(
        debug_info=no_match(["dsym"]),
        bugnumber="This test is looking explicitly for a dSYM",
    )
    @skipUnlessDarwin
    @skipIfRemote
    def test_target_arch_init(self):
        self.build()
        aout_exe = self.getBuildArtifact("a.out")
        aout_dsym = self.getBuildArtifact("a.out.dSYM")
        hidden_dir = self.getBuildArtifact("hide.noindex")
        hidden_aout_exe = self.getBuildArtifact("hide.noindex/a.out")
        hidden_aout_dsym = self.getBuildArtifact("hide.noindex/a.out.dSYM")
        dsym_for_uuid = self.getBuildArtifact("dsym-for-uuid.sh")

        # We can hook in our dsym-for-uuid shell script to lldb with
        # this env var instead of requiring a defaults write.
        os.environ["LLDB_APPLE_DSYMFORUUID_EXECUTABLE"] = dsym_for_uuid
        self.addTearDownHook(
            lambda: os.environ.pop("LLDB_APPLE_DSYMFORUUID_EXECUTABLE", None)
        )

        dwarfdump_uuid_regex = re.compile("UUID: ([-0-9a-fA-F]+) \(([^\(]+)\) .*")
        dwarfdump_cmd_output = subprocess.check_output(
            ('/usr/bin/dwarfdump --uuid "%s"' % aout_exe), shell=True
        ).decode("utf-8")
        aout_uuid = None
        for line in dwarfdump_cmd_output.splitlines():
            match = dwarfdump_uuid_regex.search(line)
            if match:
                aout_uuid = match.group(1)
        self.assertNotEqual(aout_uuid, None, "Could not get uuid of built a.out")

        ###  Create our dsym-for-uuid shell script which returns self.hidden_aout_exe.
        shell_cmds = [
            "#! /bin/sh",
            "# the last argument is the uuid",
            "while [ $# -gt 1 ]",
            "do",
            "  shift",
            "done",
            "ret=0",
            'echo "<?xml version=\\"1.0\\" encoding=\\"UTF-8\\"?>"',
            'echo "<!DOCTYPE plist PUBLIC \\"-//Apple//DTD PLIST 1.0//EN\\" \\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\\">"',
            'echo "<plist version=\\"1.0\\">"',
            "",
            'if [ "$1" = "%s" ]' % aout_uuid,
            "then",
            "  uuid=%s" % aout_uuid,
            "  bin=%s" % hidden_aout_exe,
            "  dsym=%s.dSYM/Contents/Resources/DWARF/%s"
            % (hidden_aout_exe, os.path.basename(hidden_aout_exe)),
            "fi",
            'echo "  <dict>"',
            'echo "    <key>$1</key>"',
            'echo "    <dict>"',
            'if [ -z "$uuid" -o -z "$bin" -o ! -f "$bin" ]',
            "then",
            '  echo "      <key>DBGError</key>"',
            '  echo "      <string>not found by $0</string>"',
            '  echo "    </dict>"',
            '  echo "  </dict>"',
            '  echo "</plist>"',
            "  exit 0",
            "fi",
            "",
            'echo "<key>DBGDSYMPath</key><string>$dsym</string>"',
            'echo "<key>DBGSymbolRichExecutable</key><string>$bin</string>"',
            'echo "</dict></dict></plist>"',
            "exit $ret",
        ]

        with open(dsym_for_uuid, "w") as writer:
            for l in shell_cmds:
                writer.write(l + "\n")

        os.chmod(dsym_for_uuid, 0o755)

        # Move the main binary and its dSYM into the hide.noindex
        # directory.  Now the only way lldb can find them is with
        # the LLDB_APPLE_DSYMFORUUID_EXECUTABLE shell script -
        # so we're testing that this dSYM discovery method works.
        lldbutil.mkdir_p(hidden_dir)
        os.rename(aout_exe, hidden_aout_exe)
        os.rename(aout_dsym, hidden_aout_dsym)

        target = self.dbg.CreateTarget("")
        self.assertTrue(target.IsValid())
        self.expect("target list", matching=False, substrs=["arch="])

        m = target.AddModule(None, None, aout_uuid)
        self.assertTrue(m.IsValid())
        self.expect("target list", substrs=["arch="])
