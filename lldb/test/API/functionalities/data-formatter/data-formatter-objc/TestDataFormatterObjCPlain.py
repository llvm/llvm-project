# encoding: utf-8
"""
Test lldb data formatter subsystem.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

from ObjCDataFormatterTestCase import ObjCDataFormatterTestCase


class ObjCDataFormatterNSPlain(ObjCDataFormatterTestCase):

    @skipUnlessDarwin
    def test_plain_objc_with_run_command(self):
        """Test basic ObjC formatting behavior."""
        self.build()
        self.target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, '// Set break point at this line.',
            lldb.SBFileSpec('main.m', False))

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=['stopped', 'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type synth clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd("type summary add --summary-string \"${var%@}\" MyClass")

        self.expect("frame variable object2", substrs=['MyOtherClass'])

        self.expect("frame variable *object2", substrs=['MyOtherClass'])

        # Now let's delete the 'MyClass' custom summary.
        self.runCmd("type summary delete MyClass")

        # The type format list should not show 'MyClass' at this point.
        self.expect("type summary list", matching=False, substrs=['MyClass'])

        self.runCmd("type summary add --summary-string \"a test\" MyClass")

        self.expect(
            "frame variable *object2",
            substrs=['*object2 =', 'MyClass = a test', 'backup = '])

        self.expect(
            "frame variable object2", matching=False, substrs=['a test'])

        self.expect("frame variable object", substrs=['a test'])

        self.expect("frame variable *object", substrs=['a test'])

        self.expect(
            'frame variable myclass', substrs=['(Class) myclass = NSValue'])
        self.expect(
            'frame variable myclass2',
            substrs=['(Class) myclass2 = ', 'NS', 'String'])
        self.expect(
            'frame variable myclass3', substrs=['(Class) myclass3 = Molecule'])
        self.expect(
            'frame variable myclass4',
            substrs=['(Class) myclass4 = NSMutableArray'])
        self.expect(
            'frame variable myclass5', substrs=['(Class) myclass5 = nil'])
