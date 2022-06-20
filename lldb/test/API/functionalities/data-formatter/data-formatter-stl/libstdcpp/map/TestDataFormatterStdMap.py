"""
Test lldb data formatter subsystem.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class StdMapDataFormatterTestCase(TestBase):

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    @add_test_categories(["libstdcxx"])
    @expectedFailureAll(bugnumber="llvm.org/pr50861", compiler="gcc")
    def test_with_run_command(self):
        """Test that that file and class static variables display correctly."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_source_regexp(
            self, "Set break point at this line.")

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

        self.runCmd("frame variable ii --show-types")

        self.runCmd(
            "type summary add -x \"std::map<\" --summary-string \"map has ${svar%#} items\" -e")

        self.expect('frame variable ii',
                    substrs=['map has 0 items',
                             '{}'])

        self.runCmd("c")

        self.expect('frame variable ii',
                    substrs=['map has 2 items',
                             '[0] = ',
                             'first = 0',
                             'second = 0',
                             '[1] = ',
                             'first = 1',
                             'second = 1'])

        self.runCmd("c")

        self.expect('frame variable ii',
                    substrs=['map has 4 items',
                             '[2] = ',
                             'first = 2',
                             'second = 0',
                             '[3] = ',
                             'first = 3',
                             'second = 1'])

        self.runCmd("c")

        self.expect("frame variable ii",
                    substrs=['map has 9 items',
                             '[5] = ',
                             'first = 5',
                             'second = 0',
                             '[7] = ',
                             'first = 7',
                             'second = 1'])

        self.expect("p ii",
                    substrs=['map has 9 items',
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

        self.expect("frame variable ii[8]", matching=True,
                    substrs=['1234567'])

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

        self.runCmd("c")

        self.expect('frame variable ii',
                    substrs=['map has 0 items',
                             '{}'])

        self.runCmd("frame variable si --show-types")

        self.expect('frame variable si',
                    substrs=['map has 0 items',
                             '{}'])

        self.runCmd("c")

        self.expect('frame variable si',
                    substrs=['map has 1 items',
                             '[0] = ',
                             'first = \"zero\"',
                             'second = 0'])

        self.runCmd("c")

        self.expect(
            "frame variable si",
            substrs=[
                'map has 5 items',
                '[0] = (first = "four", second = 4)',
                '[1] = (first = "one", second = 1)',
                '[2] = (first = "three", second = 3)',
                '[3] = (first = "two", second = 2)',
                '[4] = (first = "zero", second = 0)',
            ])

        self.expect(
            "p si",
            substrs=[
                'map has 5 items',
                '[0] = (first = "four", second = 4)',
                '[1] = (first = "one", second = 1)',
                '[2] = (first = "three", second = 3)',
                '[3] = (first = "two", second = 2)',
                '[4] = (first = "zero", second = 0)',
            ])

        # check access-by-index
        self.expect("frame variable si[0]",
                    substrs=['first = ', 'four',
                             'second = 4'])

        # check that MightHaveChildren() gets it right
        self.assertTrue(
            self.frame().FindVariable("si").MightHaveChildren(),
            "si.MightHaveChildren() says False for non empty!")

        # check that the expression parser does not make use of
        # synthetic children instead of running code
        # TOT clang has a fix for this, which makes the expression command here succeed
        # since this would make the test fail or succeed depending on clang version in use
        # this is safer commented for the time being
        # self.expect("expression si[0]", matching=False, error=True,
        #            substrs = ['first = ', 'zero'])

        self.runCmd("c")

        self.expect('frame variable si',
                    substrs=['map has 0 items',
                             '{}'])

        self.runCmd("frame variable is --show-types")

        self.expect('frame variable is',
                    substrs=['map has 0 items',
                             '{}'])

        self.runCmd("c")

        self.expect(
            "frame variable is",
            substrs=[
                'map has 4 items', '[0] = (first = 1, second = "is")',
                '[1] = (first = 2, second = "smart")',
                '[2] = (first = 3, second = "!!!")',
                '[3] = (first = 85, second = "goofy")'
            ])

        self.expect(
            "p is",
            substrs=[
                'map has 4 items', '[0] = (first = 1, second = "is")',
                '[1] = (first = 2, second = "smart")',
                '[2] = (first = 3, second = "!!!")',
                '[3] = (first = 85, second = "goofy")'
            ])

        # check access-by-index
        self.expect("frame variable is[0]",
                    substrs=['first = ',
                             'second ='])

        # check that MightHaveChildren() gets it right
        self.assertTrue(
            self.frame().FindVariable("is").MightHaveChildren(),
            "is.MightHaveChildren() says False for non empty!")

        # check that the expression parser does not make use of
        # synthetic children instead of running code
        # TOT clang has a fix for this, which makes the expression command here succeed
        # since this would make the test fail or succeed depending on clang version in use
        # this is safer commented for the time being
        # self.expect("expression is[0]", matching=False, error=True,
        #            substrs = ['first = ', 'goofy'])

        self.runCmd("c")

        self.expect('frame variable is',
                    substrs=['map has 0 items',
                             '{}'])

        self.runCmd("frame variable ss --show-types")

        self.expect('frame variable ss',
                    substrs=['map has 0 items',
                             '{}'])

        self.runCmd("c")

        self.expect(
            "frame variable ss",
            substrs=[
                'map has 4 items',
                '[0] = (first = "a Mac..", second = "..is always a Mac!")',
                '[1] = (first = "casa", second = "house")',
                '[2] = (first = "ciao", second = "hello")',
                '[3] = (first = "gatto", second = "cat")'
            ])

        self.expect(
            "p ss",
            substrs=[
                'map has 4 items',
                '[0] = (first = "a Mac..", second = "..is always a Mac!")',
                '[1] = (first = "casa", second = "house")',
                '[2] = (first = "ciao", second = "hello")',
                '[3] = (first = "gatto", second = "cat")'
            ])

        # check access-by-index
        self.expect("frame variable ss[3]",
                    substrs=['gatto', 'cat'])

        # check that MightHaveChildren() gets it right
        self.assertTrue(
            self.frame().FindVariable("ss").MightHaveChildren(),
            "ss.MightHaveChildren() says False for non empty!")

        # check that the expression parser does not make use of
        # synthetic children instead of running code
        # TOT clang has a fix for this, which makes the expression command here succeed
        # since this would make the test fail or succeed depending on clang version in use
        # this is safer commented for the time being
        # self.expect("expression ss[3]", matching=False, error=True,
        #            substrs = ['gatto'])

        self.runCmd("c")

        self.expect('frame variable ss',
                    substrs=['map has 0 items',
                             '{}'])
