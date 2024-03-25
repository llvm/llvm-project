"""
Test lldb data formatter subsystem.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class PrintObjectArrayTestCase(TestBase):
    @skipUnlessDarwin
    def test_print_array(self):
        """Test that expr -O -Z works"""
        self.build()
        self.printarray_data_formatter_commands()

    @skipUnlessDarwin
    def test_print_array_no_const(self):
        """Test that expr -O -Z works"""
        disable_constant_classes = {
            "CC": "xcrun clang",  # FIXME: Remove when flags are available upstream.
            "USE_SYSTEM_STDLIB": "1",  # See above.
            "CFLAGS_EXTRAS": "-fno-constant-nsnumber-literals "
            + "-fno-constant-nsarray-literals "
            + "-fno-constant-nsdictionary-literals",
        }
        self.build(dictionary=disable_constant_classes)
        self.printarray_data_formatter_commands()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number("main.mm", "break here")

    def printarray_data_formatter_commands(self):
        """Test that expr -O -Z works"""
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.mm", self.line, num_expected_locations=1, loc_exact=True
        )

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd("type format clear", check=False)
            self.runCmd("type summary clear", check=False)
            self.runCmd("type synth clear", check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.expect(
            "expr --element-count 3 --object-description -- objects",
            substrs=[
                "3735928559",
                "4276993775",
                "3203398366",
                "Hello",
                "World",
                "Two =",
                "1 =",
            ],
        )
        self.expect(
            "poarray 3 objects",
            substrs=[
                "3735928559",
                "4276993775",
                "3203398366",
                "Hello",
                "World",
                "Two =",
                "1 =",
            ],
        )
        self.expect(
            "expr --element-count 3 --object-description --description-verbosity=full -- objects",
            substrs=[
                "[0] =",
                "3735928559",
                "4276993775",
                "3203398366",
                "[1] =",
                "Hello",
                "World",
                "[2] =",
                "Two =",
                "1 =",
            ],
        )
        self.expect("parray 3 objects", substrs=["[0] = 0x", "[1] = 0x", "[2] = 0x"])
        self.expect(
            "expr --element-count 3 -d run -- objects",
            substrs=["3 elements", "2 elements", "2 key/value pairs"],
        )
        self.expect(
            "expr --element-count 3 -d run --ptr-depth=1 -- objects",
            substrs=[
                "3 elements",
                "3735928559",
                "4276993775",
                "3203398366",
                "2 elements",
                '"Hello"',
                '"World"',
                "2 key/value pairs",
            ],
        )
