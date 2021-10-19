import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCaseClassTemplateNonTypeParameterPack(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureAll(oslist=["windows"]) # Fails to read memory from target.
    @no_debug_info_test
    def test(self):
        self.build()
        self.dbg.CreateTarget(self.getBuildArtifact("a.out"))

        self.expect_expr("emptyNonTypePack", result_type="NonTypePack<>",
            result_children=[ValueCheck(name="a", type="int")])
        self.expect_expr("oneElemNonTypePack", result_type="NonTypePack<1>",
            result_children=[ValueCheck(name="a", type="int")])
        self.expect_expr("twoElemNonTypePack", result_type="NonTypePack<1, 2>",
            result_children=[ValueCheck(name="a", type="int")])


        self.expect_expr("emptyAnonNonTypePack", result_type="AnonNonTypePack<>",
            result_children=[ValueCheck(name="b", type="int")])
        self.expect_expr("oneElemAnonNonTypePack", result_type="AnonNonTypePack<1>",
            result_children=[ValueCheck(name="b", type="int")])
        self.expect_expr("twoElemAnonNonTypePack", result_type="AnonNonTypePack<1, 2>",
            result_children=[ValueCheck(name="b", type="int")])


        self.expect_expr("emptyAnonNonTypePackAfterTypeParam", result_type="AnonNonTypePackAfterTypeParam<int>",
            result_children=[ValueCheck(name="c", type="int")])
        self.expect_expr("oneElemAnonNonTypePackAfterTypeParam", result_type="AnonNonTypePackAfterTypeParam<int, 1>",
            result_children=[ValueCheck(name="c", type="int")])



        self.expect_expr("emptyAnonNonTypePackAfterAnonTypeParam", result_type="AnonNonTypePackAfterAnonTypeParam<int>",
            result_children=[ValueCheck(name="d", type="float")])
        self.expect_expr("oneElemAnonNonTypePackAfterAnonTypeParam", result_type="AnonNonTypePackAfterAnonTypeParam<int, 1>",
            result_children=[ValueCheck(name="d", type="float")])


        self.expect_expr("emptyNonTypePackAfterAnonTypeParam", result_type="NonTypePackAfterAnonTypeParam<int>",
            result_children=[ValueCheck(name="e", type="int")])
        self.expect_expr("oneElemNonTypePackAfterAnonTypeParam", result_type="NonTypePackAfterAnonTypeParam<int, 1>",
            result_children=[ValueCheck(name="e", type="int")])


        self.expect_expr("emptyNonTypePackAfterTypeParam", result_type="NonTypePackAfterTypeParam<int>",
            result_children=[ValueCheck(name="f", type="int")])
        self.expect_expr("oneElemNonTypePackAfterTypeParam", result_type="NonTypePackAfterTypeParam<int, 1>",
            result_children=[ValueCheck(name="f", type="int")])

        self.expect_expr("emptyAnonNonTypePackAfterNonTypeParam", result_type="AnonNonTypePackAfterNonTypeParam<1>",
            result_children=[ValueCheck(name="g", type="int")])
        self.expect_expr("oneElemAnonNonTypePackAfterNonTypeParam", result_type="AnonNonTypePackAfterNonTypeParam<1, 2>",
            result_children=[ValueCheck(name="g", type="int")])


        self.expect_expr("emptyAnonNonTypePackAfterAnonNonTypeParam", result_type="AnonNonTypePackAfterAnonNonTypeParam<1>",
            result_children=[ValueCheck(name="h", type="float")])
        self.expect_expr("oneElemAnonNonTypePackAfterAnonNonTypeParam", result_type="AnonNonTypePackAfterAnonNonTypeParam<1, 2>",
            result_children=[ValueCheck(name="h", type="float")])


        self.expect_expr("emptyNonTypePackAfterAnonNonTypeParam", result_type="NonTypePackAfterAnonNonTypeParam<1>",
            result_children=[ValueCheck(name="i", type="int")])
        self.expect_expr("oneElemNonTypePackAfterAnonNonTypeParam", result_type="NonTypePackAfterAnonNonTypeParam<1, 2>",
            result_children=[ValueCheck(name="i", type="int")])


        self.expect_expr("emptyNonTypePackAfterNonTypeParam", result_type="NonTypePackAfterNonTypeParam<1>",
            result_children=[ValueCheck(name="j", type="int")])
        self.expect_expr("oneElemNonTypePackAfterNonTypeParam", result_type="NonTypePackAfterNonTypeParam<1, 2>",
            result_children=[ValueCheck(name="j", type="int")])
