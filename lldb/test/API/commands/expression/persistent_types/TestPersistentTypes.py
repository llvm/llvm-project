"""
Test that lldb persistent types works correctly.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class PersistenttypesTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_persistent_types(self):
        """Test that lldb persistent types works correctly."""
        self.build()

        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        self.runCmd("breakpoint set --name main")

        self.runCmd("run", RUN_SUCCEEDED)

        self.runCmd("expression struct $foo { int a; int b; };")

        self.expect(
            "expression struct $foo $my_foo; $my_foo.a = 2; $my_foo.b = 3;",
            startstr="(int) $0 = 3")

        self.expect("expression $my_foo",
                    substrs=['a = 2', 'b = 3'])

        self.runCmd("expression typedef int $bar")

        self.expect("expression $bar i = 5; i",
                    startstr="($bar) $1 = 5")

        self.runCmd(
            "expression struct $foobar { char a; char b; char c; char d; };")
        self.runCmd("next")

        self.expect(
            "memory read foo -t $foobar",
            substrs=[
                '($foobar) 0x',
                ' = ',
                "a = 'H'",
                "b = 'e'",
                "c = 'l'",
                "d = 'l'"])  # persistent types are OK to use for memory read

        self.expect(
            "memory read foo -t $foobar -x c",
            substrs=[
                '($foobar) 0x',
                ' = ',
                "a = 'H'",
                "b = 'e'",
                "c = 'l'",
                "d = 'l'"])  # persistent types are OK to use for memory read

        self.expect(
            "memory read foo -t foobar",
            substrs=[
                '($foobar) 0x',
                ' = ',
                "a = 'H'",
                "b = 'e'",
                "c = 'l'",
                "d = 'l'"],
            matching=False,
            error=True)  # the type name is $foobar, make sure we settle for nothing less

        self.expect("expression struct { int a; int b; } x = { 2, 3 }; x",
                    substrs=['a = 2', 'b = 3'])

        self.expect(
            "expression struct { int x; int y; int z; } object; object.y = 1; object.z = 3; object.x = 2; object",
            substrs=[
                'x = 2',
                'y = 1',
                'z = 3'])

        self.expect(
            "expression struct A { int x; int y; }; struct { struct A a; int z; } object; object.a.y = 1; object.z = 3; object.a.x = 2; object",
            substrs=[
                'x = 2',
                'y = 1',
                'z = 3'])
