# encoding: utf-8
"""
Test lldb data formatter subsystem.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import datetime
import lldbutil

class ObjCDataFormatterTestCase(TestBase):

    mydir = os.path.join("functionalities", "data-formatter", "data-formatter-objc")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_plain_objc_with_dsym_and_run_command(self):
        """Test basic ObjC formatting behavior."""
        self.buildDsym()
        self.plain_data_formatter_commands()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dwarf_test
    def test_plain_objc_with_dwarf_and_run_command(self):
        """Test basic ObjC formatting behavior."""
        self.buildDwarf()
        self.plain_data_formatter_commands()

    def appkit_tester_impl(self,builder,commands):
        builder()
        self.appkit_common_data_formatters_command()
        commands()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_nsnumber_with_dsym_and_run_command(self):
        """Test formatters for NSNumber."""
        self.appkit_tester_impl(self.buildDsym,self.nsnumber_data_formatter_commands)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dwarf_test
    def test_nsnumber_with_dwarf_and_run_command(self):
        """Test formatters for NSNumber."""
        self.appkit_tester_impl(self.buildDwarf,self.nsnumber_data_formatter_commands)


    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_nsstring_with_dsym_and_run_command(self):
        """Test formatters for NSString."""
        self.appkit_tester_impl(self.buildDsym,self.nsstring_data_formatter_commands)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dwarf_test
    def test_nsstring_with_dwarf_and_run_command(self):
        """Test formatters for NSString."""
        self.appkit_tester_impl(self.buildDwarf,self.nsstring_data_formatter_commands)


    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_nscontainers_with_dsym_and_run_command(self):
        """Test formatters for NS container classes."""
        self.appkit_tester_impl(self.buildDsym,self.nscontainers_data_formatter_commands)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dwarf_test
    def test_nscontainers_with_dwarf_and_run_command(self):
        """Test formatters for  NS container classes."""
        self.appkit_tester_impl(self.buildDwarf,self.nscontainers_data_formatter_commands)


    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_nsdata_with_dsym_and_run_command(self):
        """Test formatters for NSData."""
        self.appkit_tester_impl(self.buildDsym,self.nsdata_data_formatter_commands)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dwarf_test
    def test_nsdata_with_dwarf_and_run_command(self):
        """Test formatters for  NSData."""
        self.appkit_tester_impl(self.buildDwarf,self.nsdata_data_formatter_commands)


    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_nsurl_with_dsym_and_run_command(self):
        """Test formatters for NSURL."""
        self.appkit_tester_impl(self.buildDsym,self.nsurl_data_formatter_commands)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dwarf_test
    def test_nsurl_with_dwarf_and_run_command(self):
        """Test formatters for NSURL."""
        self.appkit_tester_impl(self.buildDwarf,self.nsurl_data_formatter_commands)


    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_nserror_with_dsym_and_run_command(self):
        """Test formatters for NSError."""
        self.appkit_tester_impl(self.buildDsym,self.nserror_data_formatter_commands)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dwarf_test
    def test_nserror_with_dwarf_and_run_command(self):
        """Test formatters for NSError."""
        self.appkit_tester_impl(self.buildDwarf,self.nserror_data_formatter_commands)


    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_nsbundle_with_dsym_and_run_command(self):
        """Test formatters for NSBundle."""
        self.appkit_tester_impl(self.buildDsym,self.nsbundle_data_formatter_commands)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dwarf_test
    def test_nsbundle_with_dwarf_and_run_command(self):
        """Test formatters for NSBundle."""
        self.appkit_tester_impl(self.buildDwarf,self.nsbundle_data_formatter_commands)


    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_nsexception_with_dsym_and_run_command(self):
        """Test formatters for NSException."""
        self.appkit_tester_impl(self.buildDsym,self.nsexception_data_formatter_commands)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dwarf_test
    def test_nsexception_with_dwarf_and_run_command(self):
        """Test formatters for NSException."""
        self.appkit_tester_impl(self.buildDwarf,self.nsexception_data_formatter_commands)


    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_nsmisc_with_dsym_and_run_command(self):
        """Test formatters for misc NS classes."""
        self.appkit_tester_impl(self.buildDsym,self.nsmisc_data_formatter_commands)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dwarf_test
    def test_nsmisc_with_dwarf_and_run_command(self):
        """Test formatters for misc NS classes."""
        self.appkit_tester_impl(self.buildDwarf,self.nsmisc_data_formatter_commands)


    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_nsdate_with_dsym_and_run_command(self):
        """Test formatters for NSDate."""
        self.appkit_tester_impl(self.buildDsym,self.nsdate_data_formatter_commands)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dwarf_test
    def test_nsdate_with_dwarf_and_run_command(self):
        """Test formatters for NSDate."""
        self.appkit_tester_impl(self.buildDwarf,self.nsdate_data_formatter_commands)


    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_coreframeworks_with_dsym_and_run_command(self):
        """Test formatters for Core OSX frameworks."""
        self.buildDsym()
        self.cf_data_formatter_commands()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dwarf_test
    def test_coreframeworks_with_dwarf_and_run_command(self):
        """Test formatters for Core OSX frameworks."""
        self.buildDwarf()
        self.cf_data_formatter_commands()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_kvo_with_dsym_and_run_command(self):
        """Test the behavior of formatters when KVO is in use."""
        self.buildDsym()
        self.kvo_data_formatter_commands()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dwarf_test
    def test_kvo_with_dwarf_and_run_command(self):
        """Test the behavior of formatters when KVO is in use."""
        self.buildDwarf()
        self.kvo_data_formatter_commands()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_rdar11106605_with_dsym_and_run_command(self):
        """Check that Unicode characters come out of CFString summary correctly."""
        self.buildDsym()
        self.rdar11106605_commands()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dwarf_test
    def test_rdar11106605_with_dwarf_and_run_command(self):
        """Check that Unicode characters come out of CFString summary correctly."""
        self.buildDwarf()
        self.rdar11106605_commands()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_expr_with_dsym_and_run_command(self):
        """Test common cases of expression parser <--> formatters interaction."""
        self.buildDsym()
        self.expr_objc_data_formatter_commands()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dwarf_test
    def test_expr_with_dwarf_and_run_command(self):
        """Test common cases of expression parser <--> formatters interaction."""
        self.buildDwarf()
        self.expr_objc_data_formatter_commands()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.m', '// Set break point at this line.')

    def rdar11106605_commands(self):
        """Check that Unicode characters come out of CFString summary correctly."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type synth clear', check=False)


        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.expect('frame variable italian', substrs = ['L\'Italia è una Repubblica democratica, fondata sul lavoro. La sovranità appartiene al popolo, che la esercita nelle forme e nei limiti della Costituzione.'])
        self.expect('frame variable french', substrs = ['Que veut cette horde d\'esclaves, De traîtres, de rois conjurés?'])
        self.expect('frame variable german', substrs = ['Über-Ich und aus den Ansprüchen der sozialen Umwelt'])
        self.expect('frame variable japanese', substrs = ['色は匂へど散りぬるを'])
        self.expect('frame variable hebrew', substrs = ['לילה טוב'])


    def plain_data_formatter_commands(self):
        """Test basic ObjC formatting behavior."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type synth clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd("type summary add --summary-string \"${var%@}\" MyClass")

        self.expect("frame variable object2",
            substrs = ['MyOtherClass']);
        
        self.expect("frame variable *object2",
            substrs = ['MyOtherClass']);

        # Now let's delete the 'MyClass' custom summary.
        self.runCmd("type summary delete MyClass")

        # The type format list should not show 'MyClass' at this point.
        self.expect("type summary list", matching=False,
            substrs = ['MyClass'])

        self.runCmd("type summary add --summary-string \"a test\" MyClass")
        
        self.expect("frame variable *object2",
                    substrs = ['*object2 = {',
                               'MyClass = a test',
                               'backup = ']);
        
        self.expect("frame variable object2", matching=False,
                    substrs = ['a test']);
        
        self.expect("frame variable object",
                    substrs = ['a test']);
        
        self.expect("frame variable *object",
                    substrs = ['a test']);

        self.expect('frame variable myclass',
                    substrs = ['(Class) myclass = NSValue'])
        self.expect('frame variable myclass2',
                    substrs = ['(Class) myclass2 = __NSCFConstantString'])
        self.expect('frame variable myclass3',
                    substrs = ['(Class) myclass3 = Molecule'])
        self.expect('frame variable myclass4',
                    substrs = ['(Class) myclass4 = NSMutableArray'])
        self.expect('frame variable myclass5',
                    substrs = ['(Class) myclass5 = nil'])

    def appkit_common_data_formatters_command(self):
        """Test formatters for AppKit classes."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type synth clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)
    
    def nsnumber_data_formatter_commands(self):
        # Now enable AppKit and check we are displaying Cocoa classes correctly
        self.expect('frame variable num1 num2 num3 num4 num5 num6 num7 num8_Y num8_N num9',
                    substrs = ['(NSNumber *) num1 = ',' (int)5',
                    '(NSNumber *) num2 = ',' (float)3.1',
                    '(NSNumber *) num3 = ',' (double)3.14',
                    '(NSNumber *) num4 = ',' (long)-2',
                    '(NSNumber *) num5 = ',' (char)65',
                    '(NSNumber *) num6 = ',' (long)255',
                    '(NSNumber *) num7 = ','2000000',
                    '(NSNumber *) num8_Y = ',' @"1"',
                    '(NSNumber *) num8_N = ',' @"0"',
                    '(NSNumber *) num9 = ',' (short)-31616'])

        self.expect('frame variable decimal_one',
                    substrs = ['(NSDecimalNumber *) decimal_one = 0x','1'])

        self.expect('frame variable num_at1 num_at2 num_at3 num_at4',
                    substrs = ['(NSNumber *) num_at1 = ',' (int)12',
                    '(NSNumber *) num_at2 = ',' (int)-12',
                    '(NSNumber *) num_at3 = ',' (double)12.5',
                    '(NSNumber *) num_at4 = ',' (double)-12.5'])

    def nsstring_data_formatter_commands(self):
        self.expect('frame variable str0 str1 str2 str3 str4 str5 str6 str8 str9 str10 str11 label1 label2 processName str12',
                    substrs = ['(NSString *) str1 = ',' @"A rather short ASCII NSString object is here"',
                    '(NSString *) str0 = ',' @"255"',
                    '(NSString *) str1 = ',' @"A rather short ASCII NSString object is here"',
                    '(NSString *) str2 = ',' @"A rather short UTF8 NSString object is here"',
                    '(NSString *) str3 = ',' @"A string made with the at sign is here"',
                    '(NSString *) str4 = ',' @"This is string number 4 right here"',
                    '(NSString *) str5 = ',' @"{{1, 1}, {5, 5}}"',
                    '(NSString *) str6 = ',' @"1ST"',
                    '(NSString *) str8 = ',' @"hasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTime',
                    '(NSString *) str9 = ',' @"a very much boring task to write a string this way!!',
                    '(NSString *) str10 = ',' @"This is a Unicode string σ number 4 right here"',
                    '(NSString *) str11 = ',' @"__NSCFString"',
                    '(NSString *) label1 = ',' @"Process Name: "',
                    '(NSString *) label2 = ',' @"Process Id: "',
                    '(NSString *) str12 = ',' @"Process Name:  a.out Process Id:'])
        self.expect('frame variable attrString mutableAttrString mutableGetConst',
                    substrs = ['(NSAttributedString *) attrString = ',' @"hello world from foo"',
                    '(NSAttributedString *) mutableAttrString = ',' @"hello world from foo"',
                    '(NSString *) mutableGetConst = ',' @"foo said this string needs to be very long so much longer than whatever other string has been seen ever before by anyone of the mankind that of course this is still not long enough given what foo our friend foo our lovely dearly friend foo desired of us so i am adding more stuff here for the sake of it and for the joy of our friend who is named guess what just foo. hence, dear friend foo, stay safe, your string is now  long enough to accommodate your testing need and I will make sure that if not we extend it with even more fuzzy random meaningless words pasted one after the other from a long tiresome friday evening spent working in my office. my office mate went home but I am still randomly typing just for the fun of seeing what happens of the length of a Mutable String in Cocoa if it goes beyond one byte.. so be it, dear foo"'])

        self.expect('expr -d run-target -- path',substrs = ['usr/blah/stuff'])
        self.expect('frame variable path',substrs = ['usr/blah/stuff'])


    def nscontainers_data_formatter_commands(self):
        self.expect('frame variable newArray newDictionary newMutableDictionary cfdict_ref mutable_dict_ref cfarray_ref mutable_array_ref',
                    substrs = ['(NSArray *) newArray = ','@"50 objects"',
                    '(NSDictionary *) newDictionary = ',' 12 key/value pairs',
                    '(NSDictionary *) newMutableDictionary = ',' 21 key/value pairs',
                    '(CFDictionaryRef) cfdict_ref = ','@"3 entries"',
                    '(CFMutableDictionaryRef) mutable_dict_ref = ','@"12 entries"',
                    '(CFArrayRef) cfarray_ref = ','@"3 objects"',
                    '(CFMutableArrayRef) mutable_array_ref = ','@"11 objects"'])

        self.expect('frame variable nscounted_set',
                    substrs = ['(NSCountedSet *) nscounted_set = ','5 objects'])

        self.expect('frame variable iset1 iset2 imset',
                    substrs = ['4 indexes','512 indexes','10 indexes'])

        self.expect('frame variable mutable_bag_ref cfbag_ref binheap_ref',
                    substrs = ['(CFMutableBagRef) mutable_bag_ref = ','@"17 values"',
                    '(CFBagRef) cfbag_ref = ','@"15 values"',
                    '(CFBinaryHeapRef) binheap_ref = ','@"21 items"'])

    def nsdata_data_formatter_commands(self):
        self.expect('frame variable immutableData mutableData data_ref mutable_data_ref mutable_string_ref',
                    substrs = ['(NSData *) immutableData = ',' 4 bytes',
                    '(NSData *) mutableData = ',' 14 bytes',
                    '(CFDataRef) data_ref = ','@"5 bytes"',
                    '(CFMutableDataRef) mutable_data_ref = ','@"5 bytes"',
                    '(CFMutableStringRef) mutable_string_ref = ',' @"Wish ya knew"'])

    def nsurl_data_formatter_commands(self):
        self.expect('frame variable cfurl_ref cfchildurl_ref cfgchildurl_ref',
                    substrs = ['(CFURLRef) cfurl_ref = ','@"http://www.foo.bar',
                    'cfchildurl_ref = ','@"page.html -- http://www.foo.bar',
                    '(CFURLRef) cfgchildurl_ref = ','@"?whatever -- http://www.foo.bar/page.html"'])

        self.expect('frame variable nsurl nsurl2 nsurl3',
                    substrs = ['(NSURL *) nsurl = ','@"http://www.foo.bar',
                    '(NSURL *) nsurl2 =','@"page.html -- http://www.foo.bar',
                    '(NSURL *) nsurl3 = ','@"?whatever -- http://www.foo.bar/page.html"'])

    def nserror_data_formatter_commands(self):
        self.expect('frame variable nserror',
                    substrs = ['domain: @"Foobar" - code: 12'])

        self.expect('frame variable nserror->_userInfo',
                    substrs = ['2 key/value pairs'])

        self.expect('frame variable nserror->_userInfo --ptr-depth 1 -d run-target',
                    substrs = ['@"a"','@"b"',"1","2"])

    def nsbundle_data_formatter_commands(self):
        self.expect('frame variable bundle_string bundle_url main_bundle',
                    substrs = ['(NSBundle *) bundle_string = ',' @"/System/Library/Frameworks/Accelerate.framework"',
                    '(NSBundle *) bundle_url = ',' @"/System/Library/Frameworks/Cocoa.framework"',
                    '(NSBundle *) main_bundle = ','test/functionalities/data-formatter/data-formatter-objc'])

    def nsexception_data_formatter_commands(self):
        self.expect('frame variable except0 except1 except2 except3',
                    substrs = ['(NSException *) except0 = ','name:@"TheGuyWhoHasNoName" reason:@"cuz it\'s funny"',
                    '(NSException *) except1 = ','name:@"TheGuyWhoHasNoName~1" reason:@"cuz it\'s funny"',
                    '(NSException *) except2 = ','name:@"TheGuyWhoHasNoName`2" reason:@"cuz it\'s funny"',
                    '(NSException *) except3 = ','name:@"TheGuyWhoHasNoName/3" reason:@"cuz it\'s funny"'])

    def nsmisc_data_formatter_commands(self):
        self.expect('frame variable localhost',
                    substrs = ['<NSHost ','> localhost ((','"127.0.0.1"'])

        self.expect('frame variable my_task',
                    substrs = ['<NS','Task: 0x'])

        self.expect('frame variable range_value',
                    substrs = ['NSRange: {4, 4}'])

        self.expect('frame variable port',
                    substrs = ['(NSMachPort *) port = ',' mach port: '])

    def nsdate_data_formatter_commands(self):
        self.expect('frame variable date1 date2',
                    substrs = ['1985-04','2011-01'])

        # this test might fail if we hit the breakpoint late on December 31st of some given year
        # and midnight comes between hitting the breakpoint and running this line of code
        # hopefully the output will be revealing enough in that case :-)
        now_year = str(datetime.datetime.now().year)

        self.expect('frame variable date3 date4',
                    substrs = [now_year,'1970'])

        self.expect('frame variable date1_abs date2_abs',
                    substrs = ['1985-04','2011-01'])

        self.expect('frame variable date3_abs date4_abs',
                    substrs = [now_year,'1970'])

        self.expect('frame variable cupertino home europe',
                    substrs = ['@"America/Los_Angeles"',
                    '@"Europe/Rome"',
                    '@"Europe/Paris"'])

        self.expect('frame variable cupertino_ns home_ns europe_ns',
                    substrs = ['@"America/Los_Angeles"',
                    '@"Europe/Rome"',
                    '@"Europe/Paris"'])

        self.expect('frame variable mut_bv',
                    substrs = ['(CFMutableBitVectorRef) mut_bv = ', '1110 0110 1011 0000 1101 1010 1000 1111 0011 0101 1101 0001 00'])


    def expr_objc_data_formatter_commands(self):
        """Test common cases of expression parser <--> formatters interaction."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type synth clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        # check that the formatters are able to deal safely and correctly
        # with ValueObjects that the expression parser returns
        self.expect('expression ((id)@"Hello")', matching=False,
                    substrs = ['Hello'])

        self.expect('expression -d run -- ((id)@"Hello")',
        substrs = ['Hello'])

        self.expect('expr -d run -- label1',
            substrs = ['Process Name'])

        self.expect('expr -d run -- @"Hello"',
            substrs = ['Hello'])

        self.expect('expr -d run --object-description -- @"Hello"',
            substrs = ['Hello'])
        self.expect('expr -d run --object-description -- @"Hello"', matching=False,
            substrs = ['@"Hello" Hello'])


    def cf_data_formatter_commands(self):
        """Test formatters for Core OSX frameworks."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type synth clear', check=False)
            self.runCmd('log timers disable', check=False)


        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        # check formatters for common Objective-C types
        self.runCmd("log timers enable")
        self.expect("frame variable",
             substrs = ['(CFGregorianUnits) cf_greg_units = 1 years, 3 months, 5 days, 12 hours, 5 minutes 7 seconds',
             '(CFRange) cf_range = location=4 length=4',
             '(NSPoint) ns_point = (x=4, y=4)',
             '(NSRange) ns_range = location=4, length=4',
             '(NSRect *) ns_rect_ptr = (x=1, y=1), (width=5, height=5)',
             '(NSRect) ns_rect = (x=1, y=1), (width=5, height=5)',
             '(NSRectArray) ns_rect_arr = ((x=1, y=1), (width=5, height=5)), ...',
             '(NSSize) ns_size = (width=5, height=7)',
             '(NSSize *) ns_size_ptr = (width=5, height=7)',
             '(CGSize) cg_size = (width=1, height=6)',
             '(CGPoint) cg_point = (x=2, y=7)',
             '(CGRect) cg_rect = origin=(x=1, y=2) size=(width=7, height=7)',
             '(RGBColor) rgb_color = red=3 green=56 blue=35',
             '(RGBColor *) rgb_color_ptr = red=3 green=56 blue=35',
             '(Rect) rect = (t=4, l=8, b=4, r=7)',
             '(Rect *) rect_ptr = (t=4, l=8, b=4, r=7)',
             '(Point) point = (v=7, h=12)',
             '(Point *) point_ptr = (v=7, h=12)',
             '(HIPoint) hi_point = (x=7, y=12)',
             '(HIRect) hi_rect = origin=(x=3, y=5) size=(width=4, height=6)',
             'name:@"TheGuyWhoHasNoName" reason:@"cuz it\'s funny"',
             '1985',
             'foo_selector_impl'])
        self.runCmd('log timers dump')


    def kvo_data_formatter_commands(self):
        """Test the behavior of formatters when KVO is in use."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type synth clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        # as long as KVO is implemented by subclassing, this test should succeed
        # we should be able to dynamically figure out that the KVO implementor class
        # is a subclass of Molecule, and use the appropriate summary for it
        self.runCmd("type summary add -s JustAMoleculeHere Molecule")
        self.expect('frame variable molecule', substrs = ['JustAMoleculeHere'])
        self.runCmd("next")
        self.expect("thread list",
            substrs = ['stopped',
                       'step over'])
        self.expect('frame variable molecule', substrs = ['JustAMoleculeHere'])

        self.runCmd("next")
        # check that NSMutableDictionary's formatter is not confused when dealing with a KVO'd dictionary
        self.expect('frame variable newMutableDictionary', substrs = ['(NSDictionary *) newMutableDictionary = ',' 21 key/value pairs'])

        lldbutil.run_break_set_by_regexp (self, 'setAtoms')

        self.runCmd("continue")
        self.expect("frame variable _cmd",substrs = ['setAtoms:'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
