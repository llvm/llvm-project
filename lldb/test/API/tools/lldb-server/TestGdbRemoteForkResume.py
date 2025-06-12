from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *

from fork_testbase import GdbRemoteForkTestBase


class TestGdbRemoteForkResume(GdbRemoteForkTestBase):
    def setUp(self):
        GdbRemoteForkTestBase.setUp(self)
        if self.getPlatform() == "linux" and self.getArchitecture() in [
            "arm",
            "aarch64",
        ]:
            self.skipTest("Unsupported for Arm/AArch64 Linux")

    @add_test_categories(["fork"])
    def test_c_parent(self):
        self.resume_one_test(run_order=["parent", "parent"])

    @add_test_categories(["fork"])
    def test_c_child(self):
        self.resume_one_test(run_order=["child", "child"])

    @add_test_categories(["fork"])
    def test_c_parent_then_child(self):
        self.resume_one_test(run_order=["parent", "parent", "child", "child"])

    @add_test_categories(["fork"])
    def test_c_child_then_parent(self):
        self.resume_one_test(run_order=["child", "child", "parent", "parent"])

    @add_test_categories(["fork"])
    def test_c_interspersed(self):
        self.resume_one_test(run_order=["parent", "child", "parent", "child"])

    @add_test_categories(["fork"])
    def test_vCont_parent(self):
        self.resume_one_test(run_order=["parent", "parent"], use_vCont=True)

    @add_test_categories(["fork"])
    def test_vCont_child(self):
        self.resume_one_test(run_order=["child", "child"], use_vCont=True)

    @add_test_categories(["fork"])
    def test_vCont_parent_then_child(self):
        self.resume_one_test(
            run_order=["parent", "parent", "child", "child"], use_vCont=True
        )

    @add_test_categories(["fork"])
    def test_vCont_child_then_parent(self):
        self.resume_one_test(
            run_order=["child", "child", "parent", "parent"], use_vCont=True
        )

    @add_test_categories(["fork"])
    def test_vCont_interspersed(self):
        self.resume_one_test(
            run_order=["parent", "child", "parent", "child"], use_vCont=True
        )
