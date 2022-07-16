"""
Test using LLDB data formatters with frozen objects coming from the expression parser.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ExprFormattersTestCase(TestBase):

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.cpp.
        self.line = line_number('main.cpp',
                                '// Stop here')

    @skipIfTargetAndroid()  # skipping to avoid crashing the test runner
    @expectedFailureAndroid('llvm.org/pr24691')  # we hit an assertion in clang
    def test(self):
        """Test expr + formatters for good interoperability."""
        self.build()

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type summary clear', check=False)
            self.runCmd('type synthetic clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        """Test expr + formatters for good interoperability."""
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)
        self.runCmd("command script import formatters.py")
        self.runCmd("command script import foosynth.py")

        if self.TraceOn():
            self.runCmd("frame variable foo1 --show-types")
            self.runCmd("frame variable foo1.b --show-types")
            self.runCmd("frame variable foo1.b.b_ref --show-types")

        self.filecheck("expression --show-types -- *(new_foo(47))", __file__,
                '-check-prefix=EXPR-TYPES-NEW-FOO')
        # EXPR-TYPES-NEW-FOO: (foo) ${{.*}} = {
        # EXPR-TYPES-NEW-FOO-NEXT:   (int) a = 47
        # EXPR-TYPES-NEW-FOO-NEXT:   (int *) a_ptr = 0x
        # EXPR-TYPES-NEW-FOO-NEXT:   (bar) b = {
        # EXPR-TYPES-NEW-FOO-NEXT:     (int) i = 94
        # EXPR-TYPES-NEW-FOO-NEXT:     (int *) i_ptr = 0x
        # EXPR-TYPES-NEW-FOO-NEXT:     (baz) b = {
        # EXPR-TYPES-NEW-FOO-NEXT:       (int) h = 97
        # EXPR-TYPES-NEW-FOO-NEXT:       (int) k = 99
        # EXPR-TYPES-NEW-FOO-NEXT:     }
        # EXPR-TYPES-NEW-FOO-NEXT:     (baz &) b_ref = 0x
        # EXPR-TYPES-NEW-FOO-NEXT:   }
        # EXPR-TYPES-NEW-FOO-NEXT: }


        self.runCmd("type summary add -F formatters.foo_SummaryProvider3 foo")
        self.filecheck("expression foo1", __file__, '-check-prefix=EXPR-FOO1opts')
        # EXPR-FOO1opts: (foo) $
        # EXPR-FOO1opts-SAME: a = 12
        # EXPR-FOO1opts-SAME: a_ptr = {{[0-9]+}} -> 13
        # EXPR-FOO1opts-SAME: i = 24
        # EXPR-FOO1opts-SAME: i_ptr = {{[0-9]+}} -> 25
        # EXPR-FOO1opts-SAME: b_ref = {{[0-9]+}}
        # EXPR-FOO1opts-SAME: h = 27
        # EXPR-FOO1opts-SAME: k = 29
        # EXPR-FOO1opts-SAME: WITH_OPTS

        self.runCmd("type summary delete foo")

        self.runCmd("type summary add -F formatters.foo_SummaryProvider foo")

        self.expect("expression new_int(12)",
                    substrs=['(int *) $', ' = 0x'])

        self.runCmd(
            "type summary add -s \"${var%pointer} -> ${*var%decimal}\" \"int *\"")

        self.expect("expression new_int(12)",
                    substrs=['(int *) $', '= 0x', ' -> 12'])

        self.expect("expression foo1.a_ptr",
                    substrs=['(int *) $', '= 0x', ' -> 13'])

        self.filecheck("expression foo1", __file__, '-check-prefix=EXPR-FOO1')
        # EXPR-FOO1: (foo) $
        # EXPR-FOO1-SAME: a = 12
        # EXPR-FOO1-SAME: a_ptr = {{[0-9]+}} -> 13
        # EXPR-FOO1-SAME: i = 24
        # EXPR-FOO1-SAME: i_ptr = {{[0-9]+}} -> 25
        # EXPR-FOO1-SAME: b_ref = {{[0-9]+}}
        # EXPR-FOO1-SAME: h = 27
        # EXPR-FOO1-SAME: k = 29

        self.filecheck("expression --ptr-depth=1 -- new_foo(47)", __file__,
                '-check-prefix=EXPR-PTR-DEPTH1')
        # EXPR-PTR-DEPTH1: (foo *) $
        # EXPR-PTR-DEPTH1-SAME: a = 47
        # EXPR-PTR-DEPTH1-SAME: a_ptr = {{[0-9]+}} -> 48
        # EXPR-PTR-DEPTH1-SAME: i = 94
        # EXPR-PTR-DEPTH1-SAME: i_ptr = {{[0-9]+}} -> 95

        self.filecheck("expression foo2", __file__, '-check-prefix=EXPR-FOO2')
        # EXPR-FOO2: (foo) $
        # EXPR-FOO2-SAME: a = 121
        # EXPR-FOO2-SAME: a_ptr = {{[0-9]+}} -> 122
        # EXPR-FOO2-SAME: i = 242
        # EXPR-FOO2-SAME: i_ptr = {{[0-9]+}} -> 243
        # EXPR-FOO2-SAME: h = 245
        # EXPR-FOO2-SAME: k = 247

        object_name = self.res.GetOutput()
        object_name = object_name[7:]
        object_name = object_name[0:object_name.find(' =')]

        self.filecheck("frame variable foo2", __file__, '-check-prefix=VAR-FOO2')
        # VAR-FOO2: (foo) foo2
        # VAR-FOO2-SAME: a = 121
        # VAR-FOO2-SAME: a_ptr = {{[0-9]+}} -> 122
        # VAR-FOO2-SAME: i = 242
        # VAR-FOO2-SAME: i_ptr = {{[0-9]+}} -> 243
        # VAR-FOO2-SAME: h = 245
        # VAR-FOO2-SAME: k = 247

        # The object is the same as foo2, so use the EXPR-FOO2 checks.
        self.filecheck("expression $" + object_name, __file__,
                '-check-prefix=EXPR-FOO2')

        self.runCmd("type summary delete foo")
        self.runCmd(
            "type synthetic add --python-class foosynth.FooSyntheticProvider foo")

        self.expect("expression --show-types -- $" + object_name,
                    substrs=['(foo) $', ' = {', '(int) *i_ptr = 243'])

        self.runCmd("n")
        self.runCmd("n")

        self.runCmd("type synthetic delete foo")
        self.runCmd("type summary add -F formatters.foo_SummaryProvider foo")

        self.expect(
            "expression foo2",
            substrs=[
                '(foo) $',
                'a = 7777',
                'a_ptr = ',
                ' -> 122',
                'i = 242',
                'i_ptr = ',
                ' -> 8888'])

        self.expect("expression $" + object_name + '.a',
                    substrs=['7777'])

        self.expect("expression *$" + object_name + '.b.i_ptr',
                    substrs=['8888'])

        self.expect(
            "expression $" +
            object_name,
            substrs=[
                '(foo) $',
                'a = 121',
                'a_ptr = ',
                ' -> 122',
                'i = 242',
                'i_ptr = ',
                ' -> 8888',
                'h = 245',
                'k = 247'])

        self.runCmd("type summary delete foo")
        self.runCmd(
            "type synthetic add --python-class foosynth.FooSyntheticProvider foo")

        self.expect("expression --show-types -- $" + object_name,
                    substrs=['(foo) $', ' = {', '(int) *i_ptr = 8888'])

        self.runCmd("n")

        self.runCmd("type synthetic delete foo")
        self.runCmd("type summary add -F formatters.foo_SummaryProvider foo")

        self.expect(
            "expression $" +
            object_name,
            substrs=[
                '(foo) $',
                'a = 121',
                'a_ptr = ',
                ' -> 122',
                'i = 242',
                'i_ptr = ',
                ' -> 8888',
                'k = 247'])

        process = self.dbg.GetSelectedTarget().GetProcess()
        thread = process.GetThreadAtIndex(0)
        frame = thread.GetSelectedFrame()

        frozen = frame.EvaluateExpression("$" + object_name + ".a_ptr")

        a_data = frozen.GetPointeeData()

        error = lldb.SBError()
        self.assertTrue(
            a_data.GetUnsignedInt32(
                error,
                0) == 122,
            '*a_ptr = 122')

        ret = line_number("main.cpp", "Done initializing")
        self.runCmd("thread until " + str(ret))

        self.expect("frame variable numbers",
                    substrs=['1', '2', '3', '4', '5'])

        self.expect("expression numbers",
                    substrs=['1', '2', '3', '4', '5'])

        frozen = frame.EvaluateExpression("&numbers")

        a_data = frozen.GetPointeeData(0, 1)

        self.assertTrue(
            a_data.GetUnsignedInt32(
                error,
                0) == 1,
            'numbers[0] == 1')
        self.assertTrue(
            a_data.GetUnsignedInt32(
                error,
                4) == 2,
            'numbers[1] == 2')
        self.assertTrue(
            a_data.GetUnsignedInt32(
                error,
                8) == 3,
            'numbers[2] == 3')
        self.assertTrue(
            a_data.GetUnsignedInt32(
                error,
                12) == 4,
            'numbers[3] == 4')
        self.assertTrue(
            a_data.GetUnsignedInt32(
                error,
                16) == 5,
            'numbers[4] == 5')

        frozen = frame.EvaluateExpression("numbers")

        a_data = frozen.GetData()

        self.assertTrue(
            a_data.GetUnsignedInt32(
                error,
                0) == 1,
            'numbers[0] == 1')
        self.assertTrue(
            a_data.GetUnsignedInt32(
                error,
                4) == 2,
            'numbers[1] == 2')
        self.assertTrue(
            a_data.GetUnsignedInt32(
                error,
                8) == 3,
            'numbers[2] == 3')
        self.assertTrue(
            a_data.GetUnsignedInt32(
                error,
                12) == 4,
            'numbers[3] == 4')
        self.assertTrue(
            a_data.GetUnsignedInt32(
                error,
                16) == 5,
            'numbers[4] == 5')
