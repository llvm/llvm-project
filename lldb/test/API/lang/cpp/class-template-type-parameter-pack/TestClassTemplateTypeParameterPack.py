import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCaseClassTemplateTypeParameterPack(TestBase):

    @expectedFailureAll(oslist=["windows"], archs=["i[3-6]86", "x86_64"]) # Fails to read memory from target.
    @no_debug_info_test
    def test(self):
        self.build()
        self.dbg.CreateTarget(self.getBuildArtifact("a.out"))

        self.expect_expr("emptyTypePack", result_type="TypePack<>",
            result_children=[ValueCheck(name="a", type="int")])
        self.expect_expr("oneElemTypePack", result_type="TypePack<int>",
            result_children=[ValueCheck(name="a", type="int")])
        self.expect_expr("twoElemTypePack", result_type="TypePack<int, float>",
            result_children=[ValueCheck(name="a", type="int")])


        self.expect_expr("emptyAnonTypePack", result_type="AnonTypePack<>",
            result_children=[ValueCheck(name="b", type="int")])
        self.expect_expr("oneElemAnonTypePack", result_type="AnonTypePack<int>",
            result_children=[ValueCheck(name="b", type="int")])
        self.expect_expr("twoElemAnonTypePack", result_type="AnonTypePack<int, float>",
            result_children=[ValueCheck(name="b", type="int")])


        self.expect_expr("emptyAnonTypePackAfterTypeParam", result_type="AnonTypePackAfterTypeParam<int>",
            result_children=[ValueCheck(name="c", type="int")])
        self.expect_expr("oneElemAnonTypePackAfterTypeParam", result_type="AnonTypePackAfterTypeParam<int, float>",
            result_children=[ValueCheck(name="c", type="int")])



        self.expect_expr("emptyAnonTypePackAfterAnonTypeParam", result_type="AnonTypePackAfterAnonTypeParam<int>",
            result_children=[ValueCheck(name="d", type="float")])
        self.expect_expr("oneElemAnonTypePackAfterAnonTypeParam", result_type="AnonTypePackAfterAnonTypeParam<int, float>",
            result_children=[ValueCheck(name="d", type="float")])


        self.expect_expr("emptyTypePackAfterAnonTypeParam", result_type="TypePackAfterAnonTypeParam<int>",
            result_children=[ValueCheck(name="e", type="int")])
        self.expect_expr("oneElemTypePackAfterAnonTypeParam", result_type="TypePackAfterAnonTypeParam<int, float>",
            result_children=[ValueCheck(name="e", type="int")])


        self.expect_expr("emptyTypePackAfterTypeParam", result_type="TypePackAfterTypeParam<int>",
            result_children=[ValueCheck(name="f", type="int")])
        self.expect_expr("oneElemTypePackAfterTypeParam", result_type="TypePackAfterTypeParam<int, float>",
            result_children=[ValueCheck(name="f", type="int")])

        self.expect_expr("emptyAnonTypePackAfterNonTypeParam", result_type="AnonTypePackAfterNonTypeParam<1>",
            result_children=[ValueCheck(name="g", type="int")])
        self.expect_expr("oneElemAnonTypePackAfterNonTypeParam", result_type="AnonTypePackAfterNonTypeParam<1, int>",
            result_children=[ValueCheck(name="g", type="int")])


        self.expect_expr("emptyAnonTypePackAfterAnonNonTypeParam", result_type="AnonTypePackAfterAnonNonTypeParam<1>",
            result_children=[ValueCheck(name="h", type="float")])
        self.expect_expr("oneElemAnonTypePackAfterAnonNonTypeParam", result_type="AnonTypePackAfterAnonNonTypeParam<1, int>",
            result_children=[ValueCheck(name="h", type="float")])


        self.expect_expr("emptyTypePackAfterAnonNonTypeParam", result_type="TypePackAfterAnonNonTypeParam<1>",
            result_children=[ValueCheck(name="i", type="int")])
        self.expect_expr("oneElemTypePackAfterAnonNonTypeParam", result_type="TypePackAfterAnonNonTypeParam<1, int>",
            result_children=[ValueCheck(name="i", type="int")])


        self.expect_expr("emptyTypePackAfterNonTypeParam", result_type="TypePackAfterNonTypeParam<1>",
            result_children=[ValueCheck(name="j", type="int")])
        self.expect_expr("oneElemTypePackAfterNonTypeParam", result_type="TypePackAfterNonTypeParam<1, int>",
            result_children=[ValueCheck(name="j", type="int")])
