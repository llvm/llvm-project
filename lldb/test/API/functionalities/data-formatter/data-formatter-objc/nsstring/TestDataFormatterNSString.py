# encoding: utf-8
"""
Test lldb data formatter subsystem.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class NSStringDataFormatterTestCase(TestBase):

    def appkit_tester_impl(self, commands):
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        self.runCmd("settings set target.prefer-dynamic-value no-dynamic-values")

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type synth clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)
        commands()

    @skipUnlessDarwin
    @no_debug_info_test
    def test_nsstring_with_run_command(self):
        """Test formatters for NSString."""
        self.appkit_tester_impl(self.nsstring_data_formatter_commands)

    @skipUnlessDarwin
    @no_debug_info_test
    def test_rdar11106605_with_run_command(self):
        """Check that Unicode characters come out of CFString summary correctly."""
        self.appkit_tester_impl(self.rdar11106605_commands)

    @skipUnlessDarwin
    @no_debug_info_test
    def test_nsstring_withNULS_with_run_command(self):
        """Test formatters for NSString."""
        self.appkit_tester_impl(self.nsstring_withNULs_commands)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.m', '// break here')

    def rdar11106605_commands(self):
        """Check that Unicode characters come out of CFString summary correctly."""
        self.expect('frame variable italian', substrs=[
                    'L\'Italia è una Repubblica democratica, fondata sul lavoro. La sovranità appartiene al popolo, che la esercita nelle forme e nei limiti della Costituzione.'])
        self.expect('frame variable french', substrs=[
                    'Que veut cette horde d\'esclaves, De traîtres, de rois conjurés?'])
        self.expect('frame variable german', substrs=[
                    'Über-Ich und aus den Ansprüchen der sozialen Umwelt'])
        self.expect('frame variable japanese', substrs=['色は匂へど散りぬるを'])
        self.expect('frame variable hebrew', substrs=['לילה טוב'])

    def nsstring_data_formatter_commands(self):
        self.expect('frame variable empty str0 str1 str2 str3 str4 str5 str6 str8 str9 str10 str11 label1 label2 processName str12',
                    substrs=['(NSString *) empty = ', ' @""',
                             # '(NSString *) str0 = ',' @"255"',
                             '(NSString *) str1 = ', ' @"A rather short ASCII NSString object is here"',
                             '(NSString *) str2 = ', ' @"A rather short UTF8 NSString object is here"',
                             '(NSString *) str3 = ', ' @"A string made with the at sign is here"',
                             '(NSString *) str4 = ', ' @"This is string number 4 right here"',
                             '(NSString *) str5 = ', ' @"{{1, 1}, {5, 5}}"',
                             '(NSString *) str6 = ', ' @"1ST"',
                             '(NSString *) str8 = ', ' @"hasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTime',
                             '(NSString *) str9 = ', ' @"a very much boring task to write a string this way!!',
                             '(NSString *) str10 = ', ' @"This is a Unicode string σ number 4 right here"',
                             '(NSString *) str11 = ', ' @"__NSCFString"',
                             '(NSString *) label1 = ', ' @"Process Name: "',
                             '(NSString *) label2 = ', ' @"Process Id: "',
                             '(NSString *) str12 = ', ' @"Process Name:  a.out Process Id:'])
        self.expect(
            'frame variable attrString mutableAttrString mutableGetConst',
            substrs=[
                '(NSAttributedString *) attrString = ',
                ' @"hello world from foo"',
                '(NSAttributedString *) mutableAttrString = ',
                ' @"hello world from foo"',
                '(NSString *) mutableGetConst = ',
                ' @"foo said this string needs to be very long so much longer than whatever other string has been seen ever before by anyone of the mankind that of course this is still not long enough given what foo our friend foo our lovely dearly friend foo desired of us so i am adding more stuff here for the sake of it and for the joy of our friend who is named guess what just foo. hence, dear friend foo, stay safe, your string is now  long enough to accommodate your testing need and I will make sure that if not we extend it with even more fuzzy random meaningless words pasted one after the other from a long tiresome friday evening spent working in my office. my office mate went home but I am still randomly typing just for the fun of seeing what happens of the length of a Mutable String in Cocoa if it goes beyond one byte.. so be it, dear foo"'])

        self.expect('expr -d run-target -- path', substrs=['usr/blah/stuff'])
        self.expect('frame variable path', substrs=['usr/blah/stuff'])
        self.expect('expr -d run-target -- empty_path', substrs=['@""'])
        self.expect('frame variable empty_path', substrs=['@""'])

    def nsstring_withNULs_commands(self):
        """Check that the NSString formatter supports embedded NULs in the text"""
        self.expect(
            'po strwithNULs',
            substrs=['a very much boring task to write'])
        self.expect('expr [strwithNULs length]', substrs=['54'])
        self.expect('frame variable strwithNULs', substrs=[
                    '@"a very much boring task to write\\0a string this way!!'])
        self.expect('po strwithNULs2', substrs=[
                    'a very much boring task to write'])
        self.expect('expr [strwithNULs2 length]', substrs=['52'])
        self.expect('frame variable strwithNULs2', substrs=[
                    '@"a very much boring task to write\\0a string this way!!'])
