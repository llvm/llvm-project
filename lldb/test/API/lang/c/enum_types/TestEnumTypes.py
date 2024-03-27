"""Look up enum type information and check for correct display."""


import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *


class EnumTypesTestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number("main.c", "// Set break point at this line.")

    def test_command_line(self):
        """Test 'image lookup -t enum_test_days' and check for correct display and enum value printing."""
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// Breakpoint for bitfield", lldb.SBFileSpec("main.c")
        )

        self.expect(
            "fr var a", DATA_TYPES_DISPLAYED_CORRECTLY, patterns=[" = A\\(1\\)$"]
        )
        self.expect(
            "fr var b", DATA_TYPES_DISPLAYED_CORRECTLY, patterns=[" = B\\(2\\)$"]
        )
        self.expect(
            "fr var c", DATA_TYPES_DISPLAYED_CORRECTLY, patterns=[" = C\\(4\\)$"]
        )
        self.expect(
            "fr var ab", DATA_TYPES_DISPLAYED_CORRECTLY, patterns=[" = AB\\(3\\)$"]
        )
        self.expect("fr var ac", DATA_TYPES_DISPLAYED_CORRECTLY, patterns=[" = A | C$"])
        self.expect(
            "fr var all", DATA_TYPES_DISPLAYED_CORRECTLY, patterns=[" = ALL\\(7\\)$"]
        )
        # Test that an enum that doesn't match the heuristic we use in
        # TypeSystemClang::DumpEnumValue, gets printed as a raw integer.
        self.expect("fr var omega", DATA_TYPES_DISPLAYED_CORRECTLY, patterns=[" = 7$"])
        # Test the behavior in case have a variable of a type considered
        # 'bitfield' by the heuristic, but the value isn't actually fully
        # covered by the enumerators.
        self.expect(
            "expression (enum bitfield)nonsense",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            patterns=[" = B | C | 0x10$"],
        )

        # Break inside the main.
        bkpt_id = lldbutil.run_break_set_by_file_and_line(
            self, "main.c", self.line, num_expected_locations=1, loc_exact=True
        )
        self.runCmd("c", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        # The breakpoint should have a hit count of 1.
        lldbutil.check_breakpoint(self, bpno=1, expected_hit_count=1)

        # Look up information about the 'enum_test_days' enum type.
        # Check for correct display.
        self.expect(
            "image lookup -t enum_test_days",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=[
                "enum enum_test_days {",
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
                "kNumDays",
                "}",
            ],
        )

        enum_values = [
            "-4",
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
            "kNumDays",
            "5",
        ]

        # Make sure a pointer to an anonymous enum type does crash LLDB and displays correctly using
        # frame variable and expression commands
        self.expect(
            "frame variable f.op",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=["ops *", "f.op"],
            patterns=["0x0+$"],
        )
        self.expect(
            "frame variable *f.op",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=["ops", "*f.op", "<parent is NULL>"],
        )
        self.expect(
            "expr f.op",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=["ops *", "$"],
            patterns=["0x0+$"],
        )
        self.expect(
            "expr *f.op", DATA_TYPES_DISPLAYED_CORRECTLY, substrs=["error:"], error=True
        )

        bkpt = self.target().FindBreakpointByID(bkpt_id)
        for enum_value in enum_values:
            self.expect(
                "frame variable day",
                "check for valid enumeration value",
                substrs=[enum_value],
            )
            lldbutil.continue_to_breakpoint(self.process(), bkpt)

    def check_enum_members(self, members):
        name_matches = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
            "kNumDays",
        ]
        value_matches = [-3, -2, -1, 0, 1, 2, 3, 4]

        # First test that the list of members from the type works
        num_matches = len(name_matches)
        self.assertEqual(
            len(members),
            num_matches,
            "enum_members returns the right number of elements",
        )
        for idx in range(0, num_matches):
            member = members[idx]
            self.assertTrue(member.IsValid(), "Got a valid member for idx: %d" % (idx))
            self.assertEqual(
                member.name, name_matches[idx], "Name matches for %d" % (idx)
            )
            self.assertEqual(
                member.signed, value_matches[idx], "Value matches for %d" % (idx)
            )

    def test_api(self):
        """Test that the SBTypeEnumMember API's work correctly for enum_test_days"""
        self.build()
        target = lldbutil.run_to_breakpoint_make_target(self)

        types = target.FindTypes("enum_test_days")
        self.assertEqual(len(types), 1, "Found more than one enum_test_days type...")
        type = types.GetTypeAtIndex(0)

        # First check using the Python list returned by the type:
        self.check_enum_members(type.enum_members)

        # Now use the SBTypeEnumMemberList.
        member_list = type.GetEnumMembers()
        self.check_enum_members(member_list)

        # Now check that the by name accessor works:
        for member in member_list:
            name = member.name
            check_member = member_list[name]
            self.assertTrue(
                check_member.IsValid(), "Got a valid member for %s." % (name)
            )
            self.assertEqual(name, check_member.name, "Got back the right name")
            self.assertEqual(member.unsigned, check_member.unsigned)
