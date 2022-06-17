"""
Test some expressions involving STL data types.
"""



import unittest2
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class STLTestCase(TestBase):

    @skipIf
    @expectedFailureAll(bugnumber="llvm.org/PR36713")
    def test(self):
        """Test some expressions involving STL data types."""
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// Set break point at this line", lldb.SBFileSpec("main.cpp"))

        # Now try some expressions....

        self.runCmd(
            'expr for (int i = 0; i < hello_world.length(); ++i) { (void)printf("%c\\n", hello_world[i]); }')

        self.expect('expr associative_array.size()',
                    substrs=[' = 3'])
        self.expect('expr associative_array.count(hello_world)',
                    substrs=[' = 1'])
        self.expect('expr associative_array[hello_world]',
                    substrs=[' = 1'])
        self.expect('expr associative_array["hello"]',
                    substrs=[' = 2'])

    @expectedFailureAll(
        compiler="icc",
        bugnumber="ICC (13.1, 14-beta) do not emit DW_TAG_template_type_parameter.")
    @add_test_categories(['pyapi'])
    def test_SBType_template_aspects(self):
        """Test APIs for getting template arguments from an SBType."""
        self.build()
        (_, _, thread, _) = lldbutil.run_to_source_breakpoint(self, "// Set break point at this line", lldb.SBFileSpec("main.cpp"))
        frame0 = thread.GetFrameAtIndex(0)

        # Get the type for variable 'associative_array'.
        associative_array = frame0.FindVariable('associative_array')
        self.DebugSBValue(associative_array)
        self.assertTrue(associative_array, VALID_VARIABLE)
        map_type = associative_array.GetType()
        self.DebugSBType(map_type)
        self.assertTrue(map_type, VALID_TYPE)
        num_template_args = map_type.GetNumberOfTemplateArguments()
        self.assertTrue(num_template_args > 0)

        # We expect the template arguments to contain at least 'string' and
        # 'int'.
        expected_types = {'string': False, 'int': False}
        for i in range(num_template_args):
            t = map_type.GetTemplateArgumentType(i)
            self.DebugSBType(t)
            self.assertTrue(t, VALID_TYPE)
            name = t.GetName()
            if 'string' in name:
                expected_types['string'] = True
            elif 'int' == name:
                expected_types['int'] = True

        # Check that both entries of the dictionary have 'True' as the value.
        self.assertTrue(all(expected_types.values()))
