"""
Test lldb data formatter subsystem.
"""



import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

USE_LIBSTDCPP = "USE_LIBSTDCPP"
USE_LIBCPP = "USE_LIBCPP"

class GenericMultiMapDataFormatterTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)
        self.namespace = 'std'

    def findVariable(self, name):
        var = self.frame().FindVariable(name)
        self.assertTrue(var.IsValid())
        return var

    def getVariableType(self, name):
        var = self.findVariable(name)
        return var.GetType().GetDisplayTypeName()

    def check(self, var_name, size):
        var = self.findVariable(var_name)
        self.assertEqual(var.GetNumChildren(), size)
        children = []
        for i in range(size):
            child = var.GetChildAtIndex(i)
            children.append(ValueCheck(value=child.GetValue()))
        self.expect_var_path(var_name, type=self.getVariableType(var_name), children=children)

    def do_test_with_run_command(self, stdlib_type):
        """Test that that file and class static variables display correctly."""
        self.build(dictionary={stdlib_type: "1"})
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        bkpt = self.target().FindBreakpointByID(
            lldbutil.run_break_set_by_source_regexp(
                self, "Set break point at this line."))

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
            self.runCmd('type synth clear', check=False)
            self.runCmd(
                "settings set target.max-children-count 256",
                check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        multimap = self.namespace + "::multimap"

        # We expect that in some malformed cases the map shows size 0
        self.expect('frame variable corrupt_map',
                    substrs=[multimap, 'size=0',
                             '{}'])

        lldbutil.continue_to_breakpoint(self.process(), bkpt)

        self.expect('frame variable ii',
                    substrs=[multimap, 'size=0',
                             '{}'])

        lldbutil.continue_to_breakpoint(self.process(), bkpt)

        self.expect(
            'frame variable ii',
            substrs=[
                multimap,
                'size=2',
                '[0] = (first = 0, second = 0)',
                '[1] = (first = 1, second = 1)',
            ])

        self.check("ii", 2)

        lldbutil.continue_to_breakpoint(self.process(), bkpt)

        self.expect('frame variable ii',
                    substrs=[multimap, 'size=4',
                             '[2] = ',
                             'first = 2',
                             'second = 0',
                             '[3] = ',
                             'first = 3',
                             'second = 1'])

        self.check("ii", 4)

        lldbutil.continue_to_breakpoint(self.process(), bkpt)

        self.expect("frame variable ii",
                    substrs=[multimap, 'size=8',
                             '[5] = ',
                             'first = 5',
                             'second = 0',
                             '[7] = ',
                             'first = 7',
                             'second = 1'])

        self.check("ii", 8)

        self.expect("p ii",
                    substrs=[multimap, 'size=8',
                             '[5] = ',
                             'first = 5',
                             'second = 0',
                             '[7] = ',
                             'first = 7',
                             'second = 1'])

        # check access-by-index
        self.expect("frame variable ii[0]",
                    substrs=['first = 0',
                             'second = 0'])
        self.expect("frame variable ii[3]",
                    substrs=['first =',
                             'second ='])

        # check that MightHaveChildren() gets it right
        self.assertTrue(
            self.frame().FindVariable("ii").MightHaveChildren(),
            "ii.MightHaveChildren() says False for non empty!")

        # check that the expression parser does not make use of
        # synthetic children instead of running code
        # TOT clang has a fix for this, which makes the expression command here succeed
        # since this would make the test fail or succeed depending on clang version in use
        # this is safer commented for the time being
        # self.expect("expression ii[8]", matching=False, error=True,
        #            substrs = ['1234567'])

        lldbutil.continue_to_breakpoint(self.process(), bkpt)

        self.expect('frame variable ii',
                    substrs=[multimap, 'size=0',
                             '{}'])

        self.expect('frame variable si',
                    substrs=[multimap, 'size=0',
                             '{}'])

        lldbutil.continue_to_breakpoint(self.process(), bkpt)

        self.expect('frame variable si',
                    substrs=[multimap, 'size=1',
                             '[0] = ',
                             'first = \"zero\"',
                             'second = 0'])

        lldbutil.continue_to_breakpoint(self.process(), bkpt)

        self.expect(
            "frame variable si",
            substrs=[
                multimap,
                'size=4',
                '[0] = (first = "one", second = 1)',
                '[1] = (first = "three", second = 3)',
                '[2] = (first = "two", second = 2)',
                '[3] = (first = "zero", second = 0)',
            ])

        self.expect("p si",
                    substrs=[multimap, 'size=4',
                '[0] = (first = "one", second = 1)',
                '[1] = (first = "three", second = 3)',
                '[2] = (first = "two", second = 2)',
                '[3] = (first = "zero", second = 0)',
            ])

        # check that MightHaveChildren() gets it right
        self.assertTrue(
            self.frame().FindVariable("si").MightHaveChildren(),
            "si.MightHaveChildren() says False for non empty!")

        # check access-by-index
        self.expect("frame variable si[0]",
                    substrs=['first = ', 'one',
                             'second = 1'])

        # check that the expression parser does not make use of
        # synthetic children instead of running code
        # TOT clang has a fix for this, which makes the expression command here succeed
        # since this would make the test fail or succeed depending on clang version in use
        # this is safer commented for the time being
        # self.expect("expression si[0]", matching=False, error=True,
        #            substrs = ['first = ', 'zero'])

        lldbutil.continue_to_breakpoint(self.process(), bkpt)

        self.expect('frame variable si',
                    substrs=[multimap, 'size=0',
                             '{}'])

        lldbutil.continue_to_breakpoint(self.process(), bkpt)

        self.expect('frame variable is',
                    substrs=[multimap, 'size=0',
                             '{}'])

        lldbutil.continue_to_breakpoint(self.process(), bkpt)

        self.expect(
            "frame variable is",
            substrs=[
                multimap,
                'size=4',
                '[0] = (first = 1, second = "is")',
                '[1] = (first = 2, second = "smart")',
                '[2] = (first = 3, second = "!!!")',
                '[3] = (first = 85, second = "goofy")',
            ])

        self.expect(
            "p is",
            substrs=[
                multimap,
                'size=4',
                '[0] = (first = 1, second = "is")',
                '[1] = (first = 2, second = "smart")',
                '[2] = (first = 3, second = "!!!")',
                '[3] = (first = 85, second = "goofy")',
            ])

        # check that MightHaveChildren() gets it right
        self.assertTrue(
            self.frame().FindVariable("is").MightHaveChildren(),
            "is.MightHaveChildren() says False for non empty!")

        # check access-by-index
        self.expect("frame variable is[0]",
                    substrs=['first = ',
                             'second ='])

        # check that the expression parser does not make use of
        # synthetic children instead of running code
        # TOT clang has a fix for this, which makes the expression command here succeed
        # since this would make the test fail or succeed depending on clang version in use
        # this is safer commented for the time being
        # self.expect("expression is[0]", matching=False, error=True,
        #            substrs = ['first = ', 'goofy'])

        lldbutil.continue_to_breakpoint(self.process(), bkpt)

        self.expect('frame variable is',
                    substrs=[multimap, 'size=0',
                             '{}'])

        self.check("is", 0)

        lldbutil.continue_to_breakpoint(self.process(), bkpt)

        self.expect('frame variable ss',
                    substrs=[multimap, 'size=0',
                             '{}'])

        self.check("ss", 0)

        lldbutil.continue_to_breakpoint(self.process(), bkpt)

        self.expect(
            "frame variable ss",
            substrs=[
                multimap,
                'size=3',
                '[0] = (first = "casa", second = "house")',
                '[1] = (first = "ciao", second = "hello")',
                '[2] = (first = "gatto", second = "cat")',
            ])

        self.check("ss", 3)

        self.expect(
            "p ss",
            substrs=[
                multimap,
                'size=3',
                '[0] = (first = "casa", second = "house")',
                '[1] = (first = "ciao", second = "hello")',
                '[2] = (first = "gatto", second = "cat")',
            ])

        # check that MightHaveChildren() gets it right
        self.assertTrue(
            self.frame().FindVariable("ss").MightHaveChildren(),
            "ss.MightHaveChildren() says False for non empty!")

        # check access-by-index
        self.expect("frame variable ss[2]",
                    substrs=['gatto', 'cat'])

        # check that the expression parser does not make use of
        # synthetic children instead of running code
        # TOT clang has a fix for this, which makes the expression command here succeed
        # since this would make the test fail or succeed depending on clang version in use
        # this is safer commented for the time being
        # self.expect("expression ss[3]", matching=False, error=True,
        #            substrs = ['gatto'])

        lldbutil.continue_to_breakpoint(self.process(), bkpt)

        self.expect('frame variable ss',
                    substrs=[multimap, 'size=0',
                             '{}'])

        self.check("ss", 0)

    @add_test_categories(["libstdcxx"])
    @skipIf(compiler="clang", compiler_version=['<', '9.0'])
    def test_with_run_command_libstdcpp(self):
        self.do_test_with_run_command(USE_LIBSTDCPP)

    @skipIf(compiler="clang", compiler_version=['<', '9.0'])
    @add_test_categories(["libc++"])
    def test_with_run_command_libcpp(self):
        self.do_test_with_run_command(USE_LIBCPP)
