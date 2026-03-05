import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import TestBase
import lldbsuite.test.lldbutil as lldbutil

class TestCase(TestBase):

    @skipUnlessPlatform(["macosx"])
    @swiftTest
    def test_task_list(self):
        self.build()
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        self.runCmd("language swift task list")

        # Expected output (4 tasks, sorted by id):
        #   Task 1, addr = 0x... [awaiting Task 2] [suspended], ... Task<>() at Task.swift
        #   Task 2 'factorial-main', addr = 0x... [awaiting Task 3] [suspended], ... factorial ... at main.swift:7
        #   Task 3, addr = 0x..., ... factorial ... at main.swift:7
        #   Task 4, addr = 0x... [running], ... factorial ... at main.swift:3:5

        result = self.res.GetOutput()
        lines = [l for l in result.splitlines() if l.strip()]
        self.assertEqual(len(lines), 4)

        self.assertRegex(lines[0], r"Task 1, addr = 0x[0-9a-fA-F]+")
        self.assertIn("Task<>", lines[0])

        self.assertRegex(lines[1], r"Task 2 \('factorial-main'\), addr = 0x[0-9a-fA-F]+")
        self.assertIn("factorial", lines[1])

        self.assertRegex(lines[2], r"Task 3, addr = 0x[0-9a-fA-F]+")
        self.assertIn("factorial", lines[2])

        self.assertRegex(lines[3], r"Task 4, addr = 0x[0-9a-fA-F]+.*\[running\]")
        self.assertIn("factorial", lines[3])
        self.assertIn("main.swift:3", lines[3])
