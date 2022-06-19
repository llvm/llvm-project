"""
Test lldb data formatter subsystem.
"""



import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class SynthDataFormatterTestCase(TestBase):

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def test_with_run_command(self):
        """Test that that file and class static variables display correctly."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type filter clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        # Pick some values and check that the basics work
        self.runCmd("type filter add BagOfInts --child x --child z")
        self.expect("frame variable int_bag",
                    substrs=['x = 6',
                             'z = 8'])

        # Check we can still access the missing child by summary
        self.runCmd(
            "type summary add BagOfInts --summary-string \"y=${var.y}\"")
        self.expect('frame variable int_bag',
                    substrs=['y=7'])

        # Even if we have synth children, the summary prevails
        self.expect("frame variable int_bag", matching=False,
                    substrs=['x = 6',
                             'z = 8'])

        # if we skip synth and summary show y
        self.expect(
            "frame variable int_bag --synthetic-type false --no-summary-depth=1",
            substrs=[
                'x = 6',
                'y = 7',
                'z = 8'])

        # if we ask for raw output same happens
        self.expect("frame variable int_bag --raw-output",
                    substrs=['x = 6',
                             'y = 7',
                             'z = 8'])

        # Summary+Synth must work together
        self.runCmd(
            "type summary add BagOfInts --summary-string \"x=${var.x}\" -e")
        self.expect('frame variable int_bag',
                    substrs=['x=6',
                             'x = 6',
                             'z = 8'])

        # Same output, but using Python
        self.runCmd(
            "type summary add BagOfInts --python-script \"return 'x=%s' % valobj.GetChildMemberWithName('x').GetValue()\" -e")
        self.expect('frame variable int_bag',
                    substrs=['x=6',
                             'x = 6',
                             'z = 8'])

        # If I skip summaries, still give me the artificial children
        self.expect("frame variable int_bag --no-summary-depth=1",
                    substrs=['x = 6',
                             'z = 8'])

        # Delete synth and check that the view reflects it immediately
        self.runCmd("type filter delete BagOfInts")
        self.expect("frame variable int_bag",
                    substrs=['x = 6',
                             'y = 7',
                             'z = 8'])

        # Add the synth again and check that it's honored deeper in the
        # hierarchy
        self.runCmd("type filter add BagOfInts --child x --child z")
        self.expect('frame variable bag_bag',
                    substrs=['x = x=69 {',
                             'x = 69',
                             'z = 71',
                             'y = x=66 {',
                             'x = 66',
                             'z = 68'])
        self.expect('frame variable bag_bag', matching=False,
                    substrs=['y = 70',
                             'y = 67'])

        # Check that a synth can expand nested stuff
        self.runCmd("type filter add BagOfBags --child x.y --child y.z")
        self.expect('frame variable bag_bag',
                    substrs=['x.y = 70',
                             'y.z = 68'])

        # ...even if we get -> and . wrong
        self.runCmd("type filter add BagOfBags --child x.y --child \"y->z\"")
        self.expect('frame variable bag_bag',
                    substrs=['x.y = 70',
                             'y->z = 68'])

        # ...even bitfields
        self.runCmd(
            "type filter add BagOfBags --child x.y --child \"y->z[1-2]\"")
        self.expect('frame variable bag_bag --show-types',
                    substrs=['x.y = 70',
                             '(int:2) y->z[1-2] = 2'])

        # ...even if we format the bitfields
        self.runCmd(
            "type filter add BagOfBags --child x.y --child \"y->y[0-0]\"")
        self.runCmd("type format add \"int:1\" -f bool")
        self.expect('frame variable bag_bag --show-types',
                    substrs=['x.y = 70',
                             '(int:1) y->y[0-0] = true'])

        # ...even if we use one-liner summaries
        self.runCmd("type summary add -c BagOfBags")
        self.expect('frame variable bag_bag', substrs=[
                    '(BagOfBags) bag_bag = (x.y = 70, y->y[0-0] = true)'])

        self.runCmd("type summary delete BagOfBags")

        # now check we are dynamic (and arrays work)
        self.runCmd(
            "type filter add Plenty --child bitfield --child array[0] --child array[2]")
        self.expect('frame variable plenty_of_stuff',
                    substrs=['bitfield = 1',
                             'array[0] = 5',
                             'array[2] = 3'])

        self.runCmd("n")
        self.expect('frame variable plenty_of_stuff',
                    substrs=['bitfield = 17',
                             'array[0] = 5',
                             'array[2] = 3'])

        # skip synthetic children
        self.expect('frame variable plenty_of_stuff --synthetic-type no',
                    substrs=['some_values = 0x',
                             'array = 0x',
                             'array_size = 5'])

        # check flat printing with synthetic children
        self.expect('frame variable plenty_of_stuff --flat',
                    substrs=['plenty_of_stuff.bitfield = 17',
                             '*(plenty_of_stuff.array) = 5',
                             '*(plenty_of_stuff.array) = 3'])

        # check that we do not lose location information for our children
        self.expect('frame variable plenty_of_stuff --location',
                    substrs=['0x',
                             ':   bitfield = 17'])

        # check we work across pointer boundaries
        self.expect('frame variable plenty_of_stuff.some_values --ptr-depth=1',
                    substrs=['(BagOfInts *) plenty_of_stuff.some_values',
                             'x = 5',
                             'z = 7'])

        # but not if we don't want to
        self.runCmd("type filter add BagOfInts --child x --child z -p")
        self.expect('frame variable plenty_of_stuff.some_values --ptr-depth=1',
                    substrs=['(BagOfInts *) plenty_of_stuff.some_values',
                             'x = 5',
                             'y = 6',
                             'z = 7'])

        # check we're dynamic even if nested
        self.runCmd("type filter add BagOfBags --child x.z")
        self.expect('frame variable bag_bag',
                    substrs=['x.z = 71'])

        self.runCmd("n")
        self.expect('frame variable bag_bag',
                    substrs=['x.z = 12'])

        self.runCmd(
            'type summary add -e -s "I am always empty but have" EmptyStruct')
        self.expect('frame variable es', substrs=[
                    "I am always empty but have {}"])
        self.runCmd('type summary add -e -h -s "I am really empty" EmptyStruct')
        self.expect('frame variable es', substrs=["I am really empty"])
        self.expect(
            'frame variable es',
            substrs=["I am really empty {}"],
            matching=False)
