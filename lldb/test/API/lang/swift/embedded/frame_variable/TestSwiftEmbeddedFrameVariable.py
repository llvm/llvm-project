import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftEmbeddedFrameVariable(TestBase):
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        self.build()

        target, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        self.expect(
            "frame variable varB",
            substrs=["varB = ", "a = (field = 4.5)", "b = 123456"],
        )
        self.expect(
            "frame variable tuple",
            substrs=[
                "(a.A, a.B) tuple = {",
                "0 = (field = 4.5)",
                "1 = {",
                "a = (field = 4.5)",
                "b = 123456",
            ],
        )

        self.expect(
            "frame variable nonPayload1", substrs=["NonPayloadEnum) nonPayload1 = one"]
        )
        self.expect(
            "frame variable nonPayload2", substrs=["NonPayloadEnum) nonPayload2 = two"]
        )
        self.expect(
            "frame variable singlePayload",
            substrs=[
                "SinglePayloadEnum) singlePayload = ",
                "payload {",
                "a = (field = 4.5)",
                "b = 123456",
            ],
        )
        self.expect(
            "frame variable emptySinglePayload",
            substrs=["SinglePayloadEnum) emptySinglePayload = nonPayloadTwo"],
        )

        self.expect(
            "frame variable smallMultipayloadEnum1",
            substrs=[
                "SmallMultipayloadEnum) smallMultipayloadEnum1 = one {",
                "one = two",
            ],
        )
        self.expect(
            "frame variable smallMultipayloadEnum2",
            substrs=[
                "SmallMultipayloadEnum) smallMultipayloadEnum2 = two {",
                "two = one",
            ],
        )
        self.expect(
            "frame variable bigMultipayloadEnum1",
            substrs=[
                "BigMultipayloadEnum) bigMultipayloadEnum1 = one {",
                "0 = ",
                "(supField = 42)",
                "1 = ",
                "(supField = 43)",
                "2 = ",
                "(supField = 44)",
            ],
        )

        self.expect(
            "frame variable fullMultipayloadEnum1",
            substrs=["FullMultipayloadEnum) fullMultipayloadEnum1 = ", "(one = 120)"],
        )
        self.expect(
            "frame variable fullMultipayloadEnum2",
            substrs=[
                "FullMultipayloadEnum) fullMultipayloadEnum2 = ",
                "(two = 9.5)",
            ],
        )

        self.expect(
            "frame variable bigFullMultipayloadEnum1",
            substrs=[
                "a.BigFullMultipayloadEnum) bigFullMultipayloadEnum1 = one {",
                "one = (0 = 209, 1 = 315)",
            ],
        )
        self.expect(
            "frame variable bigFullMultipayloadEnum2",
            substrs=[
                "a.BigFullMultipayloadEnum) bigFullMultipayloadEnum2 = two {",
                "two = (0 = 452.5, 1 = 753.5)",
            ],
        )

        self.expect("frame variable sup", substrs=["Sup) sup = ", "supField = 42"])
        self.expect(
            "frame variable sub",
            substrs=[
                "Sub) sub = ",
                "Sup = {",
                "supField = 42",
                "subField = {",
                "a = (field = 4.5",
                "b = 123456",
            ],
        )
        self.expect(
            "frame variable subSub",
            substrs=[
                "SubSub) subSub =",
                "a.Sub = {",
                "a.Sup = {",
                "supField = 42",
                "subField = {",
                "a = (field = 4.5",
                "b = 123456",
                "subSubField = (field = 4.5)",
            ],
        )

        self.expect(
            "frame variable gsp",
            substrs=[
                "GenericStructPair<Int, Double>) gsp =",
                "(t = 42, u = 94.5)",
            ],
        )

        self.expect(
            "frame variable gsp2",
            substrs=[
                "a.GenericStructPair<a.Sup, a.B>) gsp2 = {",
                "t = ",
                "(supField = 42)",
                "u = {",
                "a = (field = 4.5)",
                "b = 123456",
            ],
        )

        self.expect(
            "frame variable gsp3",
            substrs=[
                "(a.GenericStructPair<a.BigFullMultipayloadEnum,",
                "a.SmallMultipayloadEnum>) gsp3 = {",
                "t = one {",
                "one = (0 = 209, 1 = 315)",
                "u = two {",
            ],
        )

        self.expect(
            "frame variable gcp",
            substrs=[
                "GenericClassPair<Double, Int>) gcp =",
                "(t = 43.799999999999997, u = 9348)",
            ],
        )

        self.expect(
            "frame variable either",
            substrs=["(a.Either<Int, Double>) either =", "left (left = 1234)"],
        )

        self.expect(
            "frame variable either2",
            substrs=[
                "(a.Either<a.Sup, a.GenericStructPair<a.BigFullMultipayloadEnum,",
                "a.SmallMultipayloadEnum>>)",
                "either2 = right {",
                "right = {",
                "t = one {",
                "one = (0 = 209, 1 = 315)",
                "u = two {",
                "two = one",
            ],
        )
