"""
Test lldb data formatter subsystem.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SmartArrayDataFormatterTestCase(TestBase):

    def test_with_run_command(self):
        """Test data formatter commands."""
        self.build()
        self.data_formatter_commands()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def data_formatter_commands(self):
        """Test that that file and class static variables display correctly."""
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        process = self.dbg.GetSelectedTarget().GetProcess()

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

# check that we are not looping here
        self.runCmd("type summary add --summary-string \"${var%V}\" SomeData")

        self.expect("frame variable data",
                    substrs=['SomeData @ 0x'])
# ${var%s}
        self.runCmd(
            "type summary add --summary-string \"ptr = ${var%s}\" \"char *\"")

        self.expect("frame variable strptr",
                    substrs=['ptr = \"',
                             'Hello world!'])

        self.expect("frame variable other.strptr",
                    substrs=['ptr = \"',
                             'Nested Hello world!'])

        self.runCmd(
            "type summary add --summary-string \"arr = ${var%s}\" -x \"char\\[[0-9]+\\]\"")

        self.expect("frame variable strarr",
                    substrs=['arr = \"',
                             'Hello world!'])

        self.expect("frame variable other.strarr",
                    substrs=['arr = \"',
                             'Nested Hello world!'])

        self.expect("expression strarr",
                    substrs=['arr = \"',
                             'Hello world!'])

        self.expect("expression other.strarr",
                    substrs=['arr = \"',
                             'Nested Hello world!'])

# ${var%c}
        self.runCmd(
            "type summary add --summary-string \"ptr = ${var%c}\" \"char *\"")

        self.expect("frame variable strptr",
                    substrs=['ptr = \"',
                             'Hello world!'])

        self.expect("frame variable other.strptr",
                    substrs=['ptr = \"',
                             'Nested Hello world!'])

        self.expect("expression strptr",
                    substrs=['ptr = \"',
                             'Hello world!'])

        self.expect("expression other.strptr",
                    substrs=['ptr = \"',
                             'Nested Hello world!'])

        self.runCmd(
            "type summary add --summary-string \"arr = ${var%c}\" -x \"char\\[[0-9]+\\]\"")

        self.expect("frame variable strarr",
                    substrs=['arr = \"',
                             'Hello world!'])

        self.expect("frame variable other.strarr",
                    substrs=['arr = \"',
                             'Nested Hello world!'])

        self.expect("expression strarr",
                    substrs=['arr = \"',
                             'Hello world!'])

        self.expect("expression other.strarr",
                    substrs=['arr = \"',
                             'Nested Hello world!'])

# ${var%char[]}
        self.runCmd(
            "type summary add --summary-string \"arr = ${var%char[]}\" -x \"char\\[[0-9]+\\]\"")

        self.expect("frame variable strarr",
                    substrs=['arr = \"',
                             'Hello world!'])

        self.expect("frame variable other.strarr",
                    substrs=['arr = ',
                             'Nested Hello world!'])

        self.expect("expression strarr",
                    substrs=['arr = \"',
                             'Hello world!'])

        self.expect("expression other.strarr",
                    substrs=['arr = ',
                             'Nested Hello world!'])

        self.runCmd(
            "type summary add --summary-string \"ptr = ${var%char[]}\" \"char *\"")

        self.expect("frame variable strptr",
                    substrs=['ptr = \"',
                             'Hello world!'])

        self.expect("frame variable other.strptr",
                    substrs=['ptr = \"',
                             'Nested Hello world!'])

        self.expect("expression strptr",
                    substrs=['ptr = \"',
                             'Hello world!'])

        self.expect("expression other.strptr",
                    substrs=['ptr = \"',
                             'Nested Hello world!'])

# ${var%a}
        self.runCmd(
            "type summary add --summary-string \"arr = ${var%a}\" -x \"char\\[[0-9]+\\]\"")

        self.expect("frame variable strarr",
                    substrs=['arr = \"',
                             'Hello world!'])

        self.expect("frame variable other.strarr",
                    substrs=['arr = ',
                             'Nested Hello world!'])

        self.expect("expression strarr",
                    substrs=['arr = \"',
                             'Hello world!'])

        self.expect("expression other.strarr",
                    substrs=['arr = ',
                             'Nested Hello world!'])

        self.runCmd(
            "type summary add --summary-string \"ptr = ${var%a}\" \"char *\"")

        self.expect("frame variable strptr",
                    substrs=['ptr = \"',
                             'Hello world!'])

        self.expect("frame variable other.strptr",
                    substrs=['ptr = \"',
                             'Nested Hello world!'])

        self.expect("expression strptr",
                    substrs=['ptr = \"',
                             'Hello world!'])

        self.expect("expression other.strptr",
                    substrs=['ptr = \"',
                             'Nested Hello world!'])

        self.runCmd(
            "type summary add --summary-string \"ptr = ${var[]%char[]}\" \"char *\"")

# I do not know the size of the data, but you are asking for a full array slice..
# use the ${var%char[]} to obtain a string as result
        self.expect("frame variable strptr", matching=False,
                    substrs=['ptr = \"',
                             'Hello world!'])

        self.expect("frame variable other.strptr", matching=False,
                    substrs=['ptr = \"',
                             'Nested Hello world!'])

        self.expect("expression strptr", matching=False,
                    substrs=['ptr = \"',
                             'Hello world!'])

        self.expect("expression other.strptr", matching=False,
                    substrs=['ptr = \"',
                             'Nested Hello world!'])

# You asked an array-style printout...
        self.runCmd(
            "type summary add --summary-string \"ptr = ${var[0-1]%char[]}\" \"char *\"")

        self.expect("frame variable strptr",
                    substrs=['ptr = ',
                             '[{H},{e}]'])

        self.expect("frame variable other.strptr",
                    substrs=['ptr = ',
                             '[{N},{e}]'])

        self.expect("expression strptr",
                    substrs=['ptr = ',
                             '[{H},{e}]'])

        self.expect("expression other.strptr",
                    substrs=['ptr = ',
                             '[{N},{e}]'])

# using [] is required here
        self.runCmd(
            "type summary add --summary-string \"arr = ${var%x}\" \"int [5]\"")

        self.expect("frame variable intarr", matching=False, substrs=[
                    '0x00000001,0x00000001,0x00000002,0x00000003,0x00000005'])

        self.expect("frame variable other.intarr", matching=False, substrs=[
                    '0x00000009,0x00000008,0x00000007,0x00000006,0x00000005'])

        self.runCmd(
            "type summary add --summary-string \"arr = ${var[]%x}\" \"int[5]\"")

        self.expect(
            "frame variable intarr",
            substrs=[
                'intarr = arr =',
                '0x00000001,0x00000001,0x00000002,0x00000003,0x00000005'])

        self.expect(
            "frame variable other.intarr",
            substrs=[
                'intarr = arr =',
                '0x00000009,0x00000008,0x00000007,0x00000006,0x00000005'])

# printing each array item as an array
        self.runCmd(
            "type summary add --summary-string \"arr = ${var[]%uint32_t[]}\" \"int[5]\"")

        self.expect(
            "frame variable intarr",
            substrs=[
                'intarr = arr =',
                '{0x00000001},{0x00000001},{0x00000002},{0x00000003},{0x00000005}'])

        self.expect(
            "frame variable other.intarr",
            substrs=[
                'intarr = arr = ',
                '{0x00000009},{0x00000008},{0x00000007},{0x00000006},{0x00000005}'])

# printing full array as an array
        self.runCmd(
            "type summary add --summary-string \"arr = ${var%uint32_t[]}\" \"int[5]\"")

        self.expect(
            "frame variable intarr",
            substrs=[
                'intarr = arr =',
                '0x00000001,0x00000001,0x00000002,0x00000003,0x00000005'])

        self.expect(
            "frame variable other.intarr",
            substrs=[
                'intarr = arr =',
                '0x00000009,0x00000008,0x00000007,0x00000006,0x00000005'])

# printing each array item as an array
        self.runCmd(
            "type summary add --summary-string \"arr = ${var[]%float32[]}\" \"float[7]\"")

        self.expect(
            "frame variable flarr",
            substrs=[
                'flarr = arr =',
                '{78.5},{77.25},{78},{76.125},{76.75},{76.875},{77}'])

        self.expect(
            "frame variable other.flarr",
            substrs=[
                'flarr = arr = ',
                '{25.5},{25.25},{25.125},{26.75},{27.375},{27.5},{26.125}'])

# printing full array as an array
        self.runCmd(
            "type summary add --summary-string \"arr = ${var%float32[]}\" \"float[7]\"")

        self.expect("frame variable flarr",
                    substrs=['flarr = arr =',
                             '78.5,77.25,78,76.125,76.75,76.875,77'])

        self.expect("frame variable other.flarr",
                    substrs=['flarr = arr =',
                             '25.5,25.25,25.125,26.75,27.375,27.5,26.125'])

# using array smart summary strings for pointers should make no sense
        self.runCmd(
            "type summary add --summary-string \"arr = ${var%float32[]}\" \"float *\"")
        self.runCmd(
            "type summary add --summary-string \"arr = ${var%int32_t[]}\" \"int *\"")

        self.expect("frame variable flptr", matching=False,
                    substrs=['78.5,77.25,78,76.125,76.75,76.875,77'])

        self.expect("frame variable intptr", matching=False,
                    substrs=['1,1,2,3,5'])

# use y and Y
        self.runCmd(
            "type summary add --summary-string \"arr = ${var%y}\" \"float[7]\"")
        self.runCmd(
            "type summary add --summary-string \"arr = ${var%y}\" \"int[5]\"")

        if process.GetByteOrder() == lldb.eByteOrderLittle:
            self.expect(
                "frame variable flarr",
                substrs=[
                    'flarr = arr =',
                    '00 00 9d 42,00 80 9a 42,00 00 9c 42,00 40 98 42,00 80 99 42,00 c0 99 42,00 00 9a 42'])
        else:
            self.expect(
                "frame variable flarr",
                substrs=[
                    'flarr = arr =',
                    '42 9d 00 00,42 9a 80 00,42 9c 00 00,42 98 40 00,42 99 80 00,42 99 c0 00,42 9a 00 00'])

        if process.GetByteOrder() == lldb.eByteOrderLittle:
            self.expect(
                "frame variable other.flarr",
                substrs=[
                    'flarr = arr =',
                    '00 00 cc 41,00 00 ca 41,00 00 c9 41,00 00 d6 41,00 00 db 41,00 00 dc 41,00 00 d1 41'])
        else:
            self.expect(
                "frame variable other.flarr",
                substrs=[
                    'flarr = arr =',
                    '41 cc 00 00,41 ca 00 00,41 c9 00 00,41 d6 00 00,41 db 00 00,41 dc 00 00,41 d1 00 00'])

        if process.GetByteOrder() == lldb.eByteOrderLittle:
            self.expect(
                "frame variable intarr",
                substrs=[
                    'intarr = arr =',
                    '01 00 00 00,01 00 00 00,02 00 00 00,03 00 00 00,05 00 00 00'])
        else:
            self.expect(
                "frame variable intarr",
                substrs=[
                    'intarr = arr =',
                    '00 00 00 01,00 00 00 01,00 00 00 02,00 00 00 03,00 00 00 05'])

        if process.GetByteOrder() == lldb.eByteOrderLittle:
            self.expect(
                "frame variable other.intarr",
                substrs=[
                    'intarr = arr = ',
                    '09 00 00 00,08 00 00 00,07 00 00 00,06 00 00 00,05 00 00 00'])
        else:
            self.expect(
                "frame variable other.intarr",
                substrs=[
                    'intarr = arr = ',
                    '00 00 00 09,00 00 00 08,00 00 00 07,00 00 00 06,00 00 00 05'])

        self.runCmd(
            "type summary add --summary-string \"arr = ${var%Y}\" \"float[7]\"")
        self.runCmd(
            "type summary add --summary-string \"arr = ${var%Y}\" \"int[5]\"")

        if process.GetByteOrder() == lldb.eByteOrderLittle:
            self.expect(
                "frame variable flarr",
                substrs=[
                    'flarr = arr =',
                    '00 00 9d 42', '00 80 9a 42', '00 00 9c 42', '00 40 98 42', '00 80 99 42', '00 c0 99 42', '00 00 9a 42'])
        else:
            self.expect(
                "frame variable flarr",
                substrs=[
                    'flarr = arr =',
                    '42 9d 00 00', '42 9a 80 00', '42 9c 00 00', '42 98 40 00', '42 99 80 00', '42 99 c0 00', '42 9a 00 00'])

        if process.GetByteOrder() == lldb.eByteOrderLittle:
            self.expect(
                "frame variable other.flarr",
                substrs=[
                    'flarr = arr =',
                    '00 00 cc 41', '00 00 ca 41', '00 00 c9 41', '00 00 d6 41', '00 00 db 41', '00 00 dc 41', '00 00 d1 41'])
        else:
            self.expect(
                "frame variable other.flarr",
                substrs=[
                    'flarr = arr =',
                    '41 cc 00 00', '41 ca 00 00', '41 c9 00 00', '41 d6 00 00', '41 db 00 00', '41 dc 00 00', '41 d1 00 00'])

        if process.GetByteOrder() == lldb.eByteOrderLittle:
            self.expect("frame variable intarr",
                        substrs=['intarr = arr =',
                                 '....,01 00 00 00',
                                 '....,05 00 00 00'])
        else:
            self.expect("frame variable intarr",
                        substrs=['intarr = arr =',
                                 '....,00 00 00 01',
                                 '....,00 00 00 05'])

        if process.GetByteOrder() == lldb.eByteOrderLittle:
            self.expect("frame variable other.intarr",
                        substrs=['intarr = arr = ',
                                 '09 00 00 00',
                                 '....,07 00 00 00'])
        else:
            self.expect("frame variable other.intarr",
                        substrs=['intarr = arr = ',
                                 '00 00 00 09',
                                 '....,00 00 00 07'])
