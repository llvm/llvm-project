import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftEmbeddedFrameVariable(TestBase):
 
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        self.build()
        self.implementation(True)

    @skipUnlessDarwin
    @swiftTest
    def test_without_ast(self):
        """Run the test turning off instantion of  Swift AST contexts in order to ensure that all type information comes from DWARF"""
        self.build()
        self.runCmd("setting set symbols.swift-enable-ast-context false")
        self.implementation(False)

    def implementation(self, ast):
        self.runCmd("setting set symbols.swift-enable-full-dwarf-debugging true")

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        frame = thread.frames[0]
        self.assertTrue(frame, "Frame 0 is valid.")

        if self.TraceOn():
            self.expect("log enable lldb types")
            self.expect("frame variable")

        alias1 = frame.FindVariable("alias1")
        field = alias1.GetChildMemberWithName("t")
        lldbutil.check_variable(self, field, False, value='1')
 
        alias2 = frame.FindVariable("alias2")
        a3 = alias2.GetChildMemberWithName("a3")
        lldbutil.check_variable(self, a3.GetChildAtIndex(0), False, value='3')
        a4 = alias2.GetChildMemberWithName("a4")
        lldbutil.check_variable(self, a4.GetChildAtIndex(0), False, value='4')
        a5 = alias2.GetChildMemberWithName("a5")
        a5t = a5.GetChildAtIndex(0)
        lldbutil.check_variable(self, a5t.GetChildAtIndex(0), False, value='5')
        lldbutil.check_variable(self, a5t.GetChildAtIndex(1), False, value='6')

        alias3 = frame.FindVariable("alias3")
        r = alias3.GetChildMemberWithName("r")
        a3 = r.GetChildMemberWithName("a3")
        lldbutil.check_variable(self, a3.GetChildAtIndex(0), False, value='3')
        a4 = r.GetChildMemberWithName("a4")
        lldbutil.check_variable(self, a4.GetChildAtIndex(0), False, value='4')
        a5 = r.GetChildMemberWithName("a5")
        a5t = a5.GetChildAtIndex(0)
        lldbutil.check_variable(self, a5t.GetChildAtIndex(0), False, value='5')
        lldbutil.check_variable(self, a5t.GetChildAtIndex(1), False, value='6')
        q1 = alias3.GetChildMemberWithName("q1")
        lldbutil.check_variable(self, q1.GetChildAtIndex(0), False, value='11')
        q2 = alias3.GetChildMemberWithName("q2")
        q2t = q2.GetChildAtIndex(0)
        lldbutil.check_variable(self, q2t.GetChildAtIndex(0), False, value='12')
        lldbutil.check_variable(self, q2t.GetChildAtIndex(1), False, value='13')

        
        array = frame.FindVariable("array")
        lldbutil.check_variable(self, array, False, num_children=4)
        for i in range(4):
            lldbutil.check_variable(self, array.GetChildAtIndex(i),
                                    False, value=str(i+1))

        varB = frame.FindVariable("varB")
        field = varB.GetChildMemberWithName("a").GetChildMemberWithName("field")
        lldbutil.check_variable(self, field, False, value="4.5")
        b = varB.GetChildMemberWithName("b")
        lldbutil.check_variable(self, b, False, value="123456")

        tuple = frame.FindVariable("tuple")
        first = tuple.GetChildAtIndex(0)
        field = first.GetChildMemberWithName("field")
        lldbutil.check_variable(self, field, False, value="4.5")
        second = tuple.GetChildAtIndex(1)
        a = second.GetChildMemberWithName("a")
        field = a.GetChildMemberWithName("field")
        lldbutil.check_variable(self, field, False, value="4.5")
        b = second.GetChildMemberWithName("b")
        lldbutil.check_variable(self, b, False, value="123456")

        nonPayload1 = frame.FindVariable("nonPayload1")
        lldbutil.check_variable(self, nonPayload1, False, value="one")

        nonPayload2 = frame.FindVariable("nonPayload2")
        lldbutil.check_variable(self, nonPayload2, False, value="two")

        singlePayload = frame.FindVariable("singlePayload")
        payload = singlePayload.GetChildMemberWithName("payload")
        field = payload.GetChildMemberWithName("a").GetChildMemberWithName("field")
        lldbutil.check_variable(self, field, False, value="4.5")
        b = payload.GetChildMemberWithName("b")
        lldbutil.check_variable(self, b, False, value="123456")

        emptySinglePayload = frame.FindVariable("emptySinglePayload")
        lldbutil.check_variable(self, emptySinglePayload, False, value="nonPayloadTwo")

        smallMultipayloadEnum1 = frame.FindVariable("smallMultipayloadEnum1")
        one = smallMultipayloadEnum1.GetChildMemberWithName("one")
        lldbutil.check_variable(self, one, False, value="two")

        smallMultipayloadEnum2 = frame.FindVariable("smallMultipayloadEnum2")
        two = smallMultipayloadEnum2.GetChildMemberWithName("two")
        lldbutil.check_variable(self, two, False, value="one")

        bigMultipayloadEnum1 = frame.FindVariable("bigMultipayloadEnum1")
        one = bigMultipayloadEnum1.GetChildMemberWithName("one")
        first = one.GetChildAtIndex(0).GetChildMemberWithName("supField")
        second = one.GetChildAtIndex(1).GetChildMemberWithName("supField")
        third = one.GetChildAtIndex(2).GetChildMemberWithName("supField")

        if False: # FIXME!
            lldbutil.check_variable(self, first, False, value="42")
            lldbutil.check_variable(self, second, False, value="43")
            lldbutil.check_variable(self, third, False, value="44")

        fullMultipayloadEnum1 = frame.FindVariable("fullMultipayloadEnum1")
        one = fullMultipayloadEnum1.GetChildMemberWithName("one")
        lldbutil.check_variable(self, one, False, value="120")

        fullMultipayloadEnum2 = frame.FindVariable("fullMultipayloadEnum2")
        two = fullMultipayloadEnum2.GetChildMemberWithName("two")
        lldbutil.check_variable(self, two, False, value="9.5")
        
        bigFullMultipayloadEnum1 = frame.FindVariable("bigFullMultipayloadEnum1")
        one = bigFullMultipayloadEnum1.GetChildMemberWithName("one")
        first = one.GetChildAtIndex(0)
        second = one.GetChildAtIndex(1)
        lldbutil.check_variable(self, first, False, value="209")
        lldbutil.check_variable(self, second, False, value="315")
        
        bigFullMultipayloadEnum2 = frame.FindVariable("bigFullMultipayloadEnum2")
        two = bigFullMultipayloadEnum2.GetChildMemberWithName("two")
        first = two.GetChildAtIndex(0)
        second = two.GetChildAtIndex(1)
        lldbutil.check_variable(self, first, False, value="452.5")
        lldbutil.check_variable(self, second, False, value="753.5")

        sup = frame.FindVariable("sup")
        supField = sup.GetChildMemberWithName("supField")
        lldbutil.check_variable(self, supField, False, value="42")

        sub = frame.FindVariable("sub")
        supField = sub.GetChildMemberWithName("supField")
        lldbutil.check_variable(self, supField, False, value="42")
        subField = sub.GetChildMemberWithName("subField")
        a = subField.GetChildMemberWithName("a")
        field = a.GetChildMemberWithName("field")
        lldbutil.check_variable(self, field, False, value="4.5")
        b = subField.GetChildMemberWithName("b")
        lldbutil.check_variable(self, b, False, value="123456")

        subSub = frame.FindVariable("subSub")
        supField = subSub.GetChildMemberWithName("supField")
        lldbutil.check_variable(self, supField, False, value="42")
        subField = subSub.GetChildMemberWithName("subField")
        a = subField.GetChildMemberWithName("a")
        field = a.GetChildMemberWithName("field")
        lldbutil.check_variable(self, field, False, value="4.5")
        b = subField.GetChildMemberWithName("b")
        lldbutil.check_variable(self, b, False, value="123456")

        subSubField = subSub.GetChildMemberWithName(
            "subSubField"
        ).GetChildMemberWithName("field")
        lldbutil.check_variable(self, subSubField, False, value="4.5")

        gsp = frame.FindVariable("gsp")
        t = gsp.GetChildMemberWithName("t")
        lldbutil.check_variable(self, t, False, value="42")
        u = gsp.GetChildMemberWithName("u")
        lldbutil.check_variable(self, u, False, value="94.5")

        gsp2 = frame.FindVariable("gsp2")
        t = gsp2.GetChildMemberWithName("t")
        supField = t.GetChildMemberWithName("supField")
        lldbutil.check_variable(self, supField, False, value="42")
        u = gsp2.GetChildMemberWithName("u")
        a = u.GetChildMemberWithName("a")
        field = a.GetChildMemberWithName("field")
        lldbutil.check_variable(self, field, False, value="4.5")
        b = u.GetChildMemberWithName("b")
        lldbutil.check_variable(self, b, False, value="123456")

        gsp3 = frame.FindVariable("gsp3")
        t = gsp3.GetChildMemberWithName("t")
        one = t.GetChildMemberWithName("one")
        first = one.GetChildAtIndex(0)
        second = one.GetChildAtIndex(1)
        lldbutil.check_variable(self, first, False, value="209")
        lldbutil.check_variable(self, second, False, value="315")
        u = gsp3.GetChildMemberWithName("u")
        two = u.GetChildMemberWithName("two")
        lldbutil.check_variable(self, two, False, value="one")

        gcp = frame.FindVariable("gcp")
        t = gcp.GetChildMemberWithName("t")
        lldbutil.check_variable(self, t, False, value="55.5")
        u = gcp.GetChildMemberWithName("u")
        lldbutil.check_variable(self, u, False, value="9348")

        either = frame.FindVariable("either")
        left = either.GetChildMemberWithName("left")
        lldbutil.check_variable(self, left, False, value="1234")

        either2 = frame.FindVariable("either2")
        right = either2.GetChildMemberWithName("right")
        t = right.GetChildMemberWithName("t")
        one = t.GetChildMemberWithName("one")
        first = one.GetChildAtIndex(0)
        second = one.GetChildAtIndex(1)
        lldbutil.check_variable(self, first, False, value="209")
        lldbutil.check_variable(self, second, False, value="315")
        u = right.GetChildMemberWithName("u")
        two = u.GetChildMemberWithName("two")
        lldbutil.check_variable(self, two, False, value='one')

        inner = frame.FindVariable("inner")
        value = inner.GetChildMemberWithName("value")
        lldbutil.check_variable(self, value, False, value='99')

        innerer = frame.FindVariable("innerer")
        innererValue = innerer.GetChildMemberWithName("innererValue")
        lldbutil.check_variable(self, innererValue, False, value='101')

        privateType = frame.FindVariable("privateType")
        privateField = privateType.GetChildMemberWithName("privateField")
        lldbutil.check_variable(self, privateField, False, value='100')

        specializedInner = frame.FindVariable("specializedInner")
        t = specializedInner.GetChildMemberWithName("t")
        lldbutil.check_variable(self, t, False, value='837')

        genericInner = frame.FindVariable("genericInner")
        t = genericInner.GetChildMemberWithName("t")
        lldbutil.check_variable(self, t, False, value='647')
        u = genericInner.GetChildMemberWithName("u")
        lldbutil.check_variable(self, u, False, value='674.5')

        functionType = frame.FindVariable("functionType")
        funcField = functionType.GetChildMemberWithName("funcField")
        lldbutil.check_variable(self, funcField, False, value='67')

        innerFunctionType = frame.FindVariable("innerFunctionType")
        innerFuncField = innerFunctionType.GetChildMemberWithName("innerFuncField")
        lldbutil.check_variable(self, innerFuncField, False, value='8479')

        inlineArray = frame.FindVariable("inlineArray")
        lldbutil.check_variable(self, inlineArray, False, num_children=4)
        for i in range(4):
            lldbutil.check_variable(self, inlineArray.GetChildAtIndex(i),
                                    False, value=str(i+1))

        string = frame.FindVariable("string")
        lldbutil.check_variable(self, string, False, summary='"Hello"')

        dictionary = frame.FindVariable("dict")
        lldbutil.check_variable(self, dictionary, False, summary='4 key/value pairs')
