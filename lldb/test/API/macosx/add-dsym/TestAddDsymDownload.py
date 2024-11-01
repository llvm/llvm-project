import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipUnlessDarwin
class AddDsymDownload(TestBase):
    dwarfdump_uuid_regex = re.compile("UUID: ([-0-9a-fA-F]+) \(([^\(]+)\) .*")

    def get_uuid(self):
        dwarfdump_cmd_output = subprocess.check_output(
            ('/usr/bin/dwarfdump --uuid "%s"' % self.exe), shell=True
        ).decode("utf-8")
        for line in dwarfdump_cmd_output.splitlines():
            match = self.dwarfdump_uuid_regex.search(line)
            if match:
                return match.group(1)
        return None

    def create_dsym_for_uuid(self):
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
            'if [ "$1" != "%s" ]' % (self.uuid),
            "then",
            '  echo "<key>DBGError</key><string>not found</string>"',
            '  echo "</plist>"',
            "  exit 1",
            "fi",
            "  uuid=%s" % self.uuid,
            "  bin=%s" % self.exe,
            "  dsym=%s" % self.dsym,
            'echo "<dict><key>$uuid</key><dict>"',
            "",
            'echo "<key>DBGDSYMPath</key><string>$dsym</string>"',
            'echo "<key>DBGSymbolRichExecutable</key><string>$bin</string>"',
            'echo "</dict></dict></plist>"',
            "exit $ret",
        ]

        with open(self.dsym_for_uuid, "w") as writer:
            for l in shell_cmds:
                writer.write(l + "\n")

        os.chmod(self.dsym_for_uuid, 0o755)

    def setUp(self):
        TestBase.setUp(self)
        self.source = "main.c"
        self.exe = self.getBuildArtifact("a.out")
        self.dsym = os.path.join(
            self.getBuildDir(),
            "hide.app/Contents/a.out.dSYM/Contents/Resources/DWARF/",
            os.path.basename(self.exe),
        )
        self.dsym_for_uuid = self.getBuildArtifact("dsym-for-uuid.sh")

        self.build(debug_info="dsym")
        self.assertTrue(os.path.exists(self.exe))
        self.assertTrue(os.path.exists(self.dsym))

        self.uuid = self.get_uuid()
        self.assertNotEqual(self.uuid, None, "Could not get uuid for a.out")

        self.create_dsym_for_uuid()

        os.environ["LLDB_APPLE_DSYMFORUUID_EXECUTABLE"] = self.dsym_for_uuid
        self.addTearDownHook(
            lambda: os.environ.pop("LLDB_APPLE_DSYMFORUUID_EXECUTABLE", None)
        )

    def do_test(self, command):
        self.target = self.dbg.CreateTarget(self.exe)
        self.assertTrue(self.target, VALID_TARGET)

        main_bp = self.target.BreakpointCreateByName("main", "a.out")
        self.assertTrue(main_bp, VALID_BREAKPOINT)

        self.process = self.target.LaunchSimple(
            None, None, self.get_process_working_directory()
        )
        self.assertTrue(self.process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        self.assertState(
            self.process.GetState(), lldb.eStateStopped, STOPPED_DUE_TO_BREAKPOINT
        )

        self.runCmd(command)
        self.expect("frame select", substrs=["a.out`main at main.c"])

    @no_debug_info_test
    def test_frame(self):
        self.do_test("add-dsym --frame")

    @no_debug_info_test
    def test_uuid(self):
        self.do_test("add-dsym --uuid {}".format(self.uuid))

    @no_debug_info_test
    def test_stack(self):
        self.do_test("add-dsym --stack")
