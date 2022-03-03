import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest

class TestIOHandlerProcessSTDIO(PExpectTest):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    # PExpect uses many timeouts internally and doesn't play well
    # under ASAN on a loaded machine..
    @skipIfAsan
    def test(self):
        self.build()
        self.launch(executable=self.getBuildArtifact("a.out"))
        self.child.sendline("run")

        self.child.send("foo\n")
        self.child.expect_exact("stdout: foo")

        self.child.send("bar\n")
        self.child.expect_exact("stdout: bar")

        self.child.send("baz\n")
        self.child.expect_exact("stdout: baz")

        self.child.sendcontrol('d')
        self.quit()
