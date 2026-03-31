import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import shutil
import os


class TestCTF(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def no_ctf_convert(self):
        if not shutil.which("ctfconvert"):
            return "ctfconvert not found in path"
        return None

    def no_objcopy(self):
        if not "OBJCOPY" in os.environ:
            return "llvm-objcopy not found in environment"
        return None

    @skipTestIfFn(no_ctf_convert)
    @skipTestIfFn(no_objcopy)
    @skipUnlessDarwin
    def test(self):
        self.build()
        self.do_test()

    @skipTestIfFn(no_ctf_convert)
    @skipTestIfFn(no_objcopy)
    @skipUnlessDarwin
    def test_compressed(self):
        self.build(dictionary={"COMPRESS_CTF": "YES"})
        self.do_test()

    def do_test(self):
        lldbutil.run_to_name_breakpoint(self, "printf")

        symbol_file = self.getBuildArtifact("a.ctf")

        if self.TraceOn():
            self.runCmd("log enable -v lldb symbol")

        self.runCmd("target symbols add {}".format(symbol_file))
        self.expect(
            "target variable foo",
            substrs=[
                "(MyStructT) foo",
                "i = 1",
                "foo",
                "'c'",
                "[0] = 'c'",
                "[1] = 'a'",
                "[2] = 'b'",
                "[3] = 'c'",
                'u = (i = 1, s = "")',
                "b = false",
                "f = 0x0000000000000000",
            ],
        )
        self.expect("target variable foo.n.i", substrs=["(MyInt) foo.n.i = 1"])
        self.expect(
            "target variable foo.n.s", substrs=["(const char *) foo.n.s", '"foo"']
        )
        self.expect(
            "target variable foo.n.c", substrs=["(volatile char) foo.n.c = 'c'"]
        )
        self.expect(
            "target variable foo.n.a",
            substrs=[
                "(char[4]:8) foo.n.a",
                "[0] = 'c'",
                "[1] = 'a'",
                "[2] = 'b'",
                "[3] = 'c'",
            ],
        )
        self.expect(
            "target variable foo.n.u", substrs=['(MyUnionT) foo.n.u = (i = 1, s = "")']
        )
        self.expect(
            "target variable foo.f",
            substrs=["(void (*)(int)) foo.f = 0x0000000000000000"],
        )

        self.expect(
            "type lookup MyEnum",
            substrs=[
                "enum MyEnum {",
                "eOne,",
                "eTwo,",
                "eThree",
                "}",
            ],
        )

        self.expect("type lookup RecursiveStruct", substrs=["RecursiveStruct *n;"])
