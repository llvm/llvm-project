# RUN: %PYTHON %s 2>&1 | FileCheck %s

from mlir.ir import *
from mlir.dialects.irdl import dsl as irdsl
from mlir.dialects import arith
import sys


def run(f):
    print("\nTEST:", f.__name__, file=sys.stderr)
    f()


# CHECK: TEST: testMyInt
@run
def testMyInt():
    myint = irdsl.Dialect("myint")
    iattr = irdsl.BaseName("#builtin.integer")
    i32 = irdsl.Is[IntegerType.get_signless](32)

    class ConstantOp(myint.Operation, name="constant"):
        value = irdsl.Attribute(iattr)
        cst = irdsl.Result(i32)

    class AddOp(myint.Operation, name="add"):
        lhs = irdsl.Operand(i32)
        rhs = irdsl.Operand(i32)
        res = irdsl.Result(i32)

    # CHECK: irdl.dialect @myint {
    # CHECK:   irdl.operation @constant {
    # CHECK:     %0 = irdl.base "#builtin.integer"
    # CHECK:     irdl.attributes {"value" = %0}
    # CHECK:     %1 = irdl.is i32
    # CHECK:     irdl.results(cst: %1)
    # CHECK:   }
    # CHECK:   irdl.operation @add {
    # CHECK:     %0 = irdl.is i32
    # CHECK:     irdl.operands(lhs: %0, rhs: %0)
    # CHECK:     irdl.results(res: %0)
    # CHECK:   }
    # CHECK: }
    with Context(), Location.unknown():
        myint.load()
        print(myint.mlir_module)

        # CHECK: ['constant', 'add']
        print([i._op_name for i in myint.operations])

        i32 = IntegerType.get_signless(32)

        module = Module.create()
        with InsertionPoint(module.body):
            two = ConstantOp(i32, IntegerAttr.get(i32, 2))
            three = ConstantOp(i32, IntegerAttr.get(i32, 3))
            add1 = AddOp(i32, two, three)
            add2 = AddOp(i32, add1, two)
            add3 = AddOp(i32, add2, three)

        # CHECK: %0 = "myint.constant"() {value = 2 : i32} : () -> i32
        # CHECK: %1 = "myint.constant"() {value = 3 : i32} : () -> i32
        # CHECK: %2 = "myint.add"(%0, %1) : (i32, i32) -> i32
        # CHECK: %3 = "myint.add"(%2, %0) : (i32, i32) -> i32
        # CHECK: %4 = "myint.add"(%3, %1) : (i32, i32) -> i32
        print(module)
        assert module.operation.verify()

        # CHECK: AddOp
        print(type(add1).__name__)
        # CHECK: ConstantOp
        print(type(two).__name__)
        # CHECK: myint.add
        print(add1.OPERATION_NAME)
        # CHECK: None
        print(add1._ODS_OPERAND_SEGMENTS)
        # CHECK: None
        print(add1._ODS_RESULT_SEGMENTS)
        # CHECK: %0 = "myint.constant"() {value = 2 : i32} : () -> i32
        print(add1.lhs.owner)
        # CHECK: %1 = "myint.constant"() {value = 3 : i32} : () -> i32
        print(add1.rhs.owner)
        # CHECK: 2 : i32
        print(two.value)
        # CHECK: Value(%0
        print(two.cst)
        # CHECK: (self, /, res, lhs, rhs, *, loc=None, ip=None)
        print(AddOp.__init__.__signature__)
        # CHECK: (self, /, cst, value, *, loc=None, ip=None)
        print(ConstantOp.__init__.__signature__)


@run
def testIRDSL():
    test = irdsl.Dialect("irdsl_test")
    i32 = irdsl.Is[IntegerType.get_signless](32)
    i64 = irdsl.Is[IntegerType.get_signless](64)
    i32or64 = i32 | i64
    any = irdsl.Any()
    f32 = irdsl.Is[F32Type]
    iattr = irdsl.BaseName("#builtin.integer")
    fattr = irdsl.BaseName("#builtin.float")

    class ConstraintOp(test.Operation, name="constraint"):
        a = irdsl.Operand(i32or64)
        b = irdsl.Operand(any)
        c = irdsl.Operand(f32 | i32)
        d = irdsl.Operand(any)
        x = irdsl.Attribute(iattr)
        y = irdsl.Attribute(fattr)

    class OptionalOp(test.Operation, name="optional"):
        a = irdsl.Operand(i32)
        b = irdsl.Operand(i32, irdsl.Variadicity.optional)
        out1 = irdsl.Result(i32)
        out2 = irdsl.Result(i32, irdsl.Variadicity.optional)
        out3 = irdsl.Result(i32)

    class Optional2Op(test.Operation, name="optional2"):
        a = irdsl.Operand(i32, irdsl.Variadicity.optional)
        b = irdsl.Result(i32, irdsl.Variadicity.optional)

    class VariadicOp(test.Operation, name="variadic"):
        a = irdsl.Operand(i32)
        b = irdsl.Operand(i32, irdsl.Variadicity.optional)
        c = irdsl.Operand(i32, irdsl.Variadicity.variadic)
        out1 = irdsl.Result(i32, irdsl.Variadicity.variadic)
        out2 = irdsl.Result(i32, irdsl.Variadicity.variadic)
        out3 = irdsl.Result(i32, irdsl.Variadicity.optional)
        out4 = irdsl.Result(i32)

    class Variadic2Op(test.Operation, name="variadic2"):
        a = irdsl.Operand(i32, irdsl.Variadicity.variadic)
        b = irdsl.Result(i32, irdsl.Variadicity.variadic)

    class MixedOp(test.Operation, name="mixed"):
        out = irdsl.Result(i32)
        in1 = irdsl.Operand(i32)
        in2 = irdsl.Attribute(iattr)
        in3 = irdsl.Operand(i32, irdsl.Variadicity.optional)
        in4 = irdsl.Attribute(iattr)
        in5 = irdsl.Operand(i32)

    # CHECK: irdl.dialect @irdsl_test {
    # CHECK:   irdl.operation @constraint {
    # CHECK:     %0 = irdl.is i32
    # CHECK:     %1 = irdl.is i64
    # CHECK:     %2 = irdl.any_of(%0, %1)
    # CHECK:     %3 = irdl.any
    # CHECK:     %4 = irdl.is f32
    # CHECK:     %5 = irdl.any_of(%4, %0)
    # CHECK:     irdl.operands(a: %2, b: %3, c: %5, d: %3)
    # CHECK:     %6 = irdl.base "#builtin.integer"
    # CHECK:     %7 = irdl.base "#builtin.float"
    # CHECK:     irdl.attributes {"x" = %6, "y" = %7}
    # CHECK:   }
    # CHECK:   irdl.operation @optional {
    # CHECK:     %0 = irdl.is i32
    # CHECK:     irdl.operands(a: %0, b: optional %0)
    # CHECK:     irdl.results(out1: %0, out2: optional %0, out3: %0)
    # CHECK:   }
    # CHECK:   irdl.operation @optional2 {
    # CHECK:     %0 = irdl.is i32
    # CHECK:     irdl.operands(a: optional %0)
    # CHECK:     irdl.results(b: optional %0)
    # CHECK:   }
    # CHECK:   irdl.operation @variadic {
    # CHECK:     %0 = irdl.is i32
    # CHECK:     irdl.operands(a: %0, b: optional %0, c: variadic %0)
    # CHECK:     irdl.results(out1: variadic %0, out2: variadic %0, out3: optional %0, out4: %0)
    # CHECK:   }
    # CHECK:   irdl.operation @variadic2 {
    # CHECK:     %0 = irdl.is i32
    # CHECK:     irdl.operands(a: variadic %0)
    # CHECK:     irdl.results(b: variadic %0)
    # CHECK:   }
    # CHECK:   irdl.operation @mixed {
    # CHECK:     %0 = irdl.is i32
    # CHECK:     irdl.operands(in1: %0, in3: optional %0, in5: %0)
    # CHECK:     %1 = irdl.base "#builtin.integer"
    # CHECK:     irdl.attributes {"in2" = %1, "in4" = %1}
    # CHECK:     irdl.results(out: %0)
    # CHECK:   }
    # CHECK: }
    with Context(), Location.unknown():
        test.load()
        print(test.mlir_module)

        # CHECK: (self, /, a, b, c, d, x, y, *, loc=None, ip=None)
        print(ConstraintOp.__init__.__signature__)
        # CHECK: (self, /, out1, out3, a, *, out2=None, b=None, loc=None, ip=None)
        print(OptionalOp.__init__.__signature__)
        # CHECK: (self, /, *, b=None, a=None, loc=None, ip=None)
        print(Optional2Op.__init__.__signature__)
        # CHECK: (self, /, out1, out2, out4, a, c, *, out3=None, b=None, loc=None, ip=None)
        print(VariadicOp.__init__.__signature__)
        # CHECK: (self, /, b, a, *, loc=None, ip=None)
        print(Variadic2Op.__init__.__signature__)
        # CHECK: (self, /, out, in1, in2, in4, in5, *, in3=None, loc=None, ip=None)
        print(MixedOp.__init__.__signature__)

        # CHECK: None None
        print(ConstraintOp._ODS_OPERAND_SEGMENTS, ConstraintOp._ODS_RESULT_SEGMENTS)
        # CHECK: [1, 0] [1, 0, 1]
        print(OptionalOp._ODS_OPERAND_SEGMENTS, OptionalOp._ODS_RESULT_SEGMENTS)
        # CHECK: [0] [0]
        print(Optional2Op._ODS_OPERAND_SEGMENTS, Optional2Op._ODS_RESULT_SEGMENTS)
        # CHECK: [1, 0, -1] [-1, -1, 0, 1]
        print(VariadicOp._ODS_OPERAND_SEGMENTS, VariadicOp._ODS_RESULT_SEGMENTS)
        # CHECK: [-1] [-1]
        print(Variadic2Op._ODS_OPERAND_SEGMENTS, Variadic2Op._ODS_RESULT_SEGMENTS)

        i32 = IntegerType.get_signless(32)
        i64 = IntegerType.get_signless(64)
        f32 = F32Type.get()

        iattr = IntegerAttr.get(i32, 2)
        fattr = FloatAttr.get_f32(2.3)

        module = Module.create()
        with InsertionPoint(module.body):
            ione = arith.constant(i32, 1)
            fone = arith.constant(f32, 1.2)

            # CHECK: "irdsl_test.constraint"(%c1_i32, %c1_i32, %cst, %c1_i32) {x = 2 : i32, y = 2.300000e+00 : f32} : (i32, i32, f32, i32) -> ()
            c1 = ConstraintOp(ione, ione, fone, ione, iattr, fattr)
            # CHECK: "irdsl_test.constraint"(%c1_i32, %cst, %cst, %cst) {x = 2 : i32, y = 2.300000e+00 : f32} : (i32, f32, f32, f32) -> ()
            ConstraintOp(ione, fone, fone, fone, iattr, fattr)
            # CHECK: irdsl_test.constraint"(%c1_i32, %cst, %c1_i32, %cst) {x = 2 : i32, y = 2.300000e+00 : f32} : (i32, f32, i32, f32) -> ()
            ConstraintOp(ione, fone, ione, fone, iattr, fattr)

            # CHECK: %0:2 = "irdsl_test.optional"(%c1_i32) {operandSegmentSizes = array<i32: 1, 0>, resultSegmentSizes = array<i32: 1, 0, 1>} : (i32) -> (i32, i32)
            o1 = OptionalOp(i32, i32, ione)
            # CHECK: %1:3 = "irdsl_test.optional"(%c1_i32, %c1_i32) {operandSegmentSizes = array<i32: 1, 1>, resultSegmentSizes = array<i32: 1, 1, 1>} : (i32, i32) -> (i32, i32, i32)
            o2 = OptionalOp(i32, i32, ione, out2=i32, b=ione)
            # CHECK: irdsl_test.optional2"() {operandSegmentSizes = array<i32: 0>, resultSegmentSizes = array<i32: 0>} : () -> ()
            o3 = Optional2Op()
            # CHECK: %2 = "irdsl_test.optional2"() {operandSegmentSizes = array<i32: 0>, resultSegmentSizes = array<i32: 1>} : () -> i32
            o4 = Optional2Op(b=i32)
            # CHECK: "irdsl_test.optional2"(%c1_i32) {operandSegmentSizes = array<i32: 1>, resultSegmentSizes = array<i32: 0>} : (i32) -> ()
            o5 = Optional2Op(a=ione)
            # CHECK: %3 = "irdsl_test.optional2"(%c1_i32) {operandSegmentSizes = array<i32: 1>, resultSegmentSizes = array<i32: 1>} : (i32) -> i32
            o6 = Optional2Op(b=i32, a=ione)

            # CHECK: %4:4 = "irdsl_test.variadic"(%c1_i32, %c1_i32, %c1_i32) {operandSegmentSizes = array<i32: 1, 0, 2>, resultSegmentSizes = array<i32: 1, 2, 0, 1>} : (i32, i32, i32) -> (i32, i32, i32, i32)
            v1 = VariadicOp([i32], [i32, i32], i32, ione, [ione, ione])
            # CHECK: %5:5 = "irdsl_test.variadic"(%c1_i32, %c1_i32, %c1_i32) {operandSegmentSizes = array<i32: 1, 1, 1>, resultSegmentSizes = array<i32: 1, 2, 1, 1>} : (i32, i32, i32) -> (i32, i32, i32, i32, i32)
            v2 = VariadicOp([i32], [i32, i32], i32, ione, [ione], out3=i32, b=ione)
            # CHECK: %6:4 = "irdsl_test.variadic"(%c1_i32) {operandSegmentSizes = array<i32: 1, 0, 0>, resultSegmentSizes = array<i32: 2, 1, 0, 1>} : (i32) -> (i32, i32, i32, i32)
            v3 = VariadicOp([i32, i32], [i32], i32, ione, [])
            # CHECK: "irdsl_test.variadic2"() {operandSegmentSizes = array<i32: 0>, resultSegmentSizes = array<i32: 0>} : () -> ()
            v4 = Variadic2Op([], [])
            # CHECK: "irdsl_test.variadic2"(%c1_i32, %c1_i32, %c1_i32) {operandSegmentSizes = array<i32: 3>, resultSegmentSizes = array<i32: 0>} : (i32, i32, i32) -> ()
            v5 = Variadic2Op([], [ione, ione, ione])
            # CHECK: %7:2 = "irdsl_test.variadic2"(%c1_i32) {operandSegmentSizes = array<i32: 1>, resultSegmentSizes = array<i32: 2>} : (i32) -> (i32, i32)
            v6 = Variadic2Op([i32, i32], [ione])

            # CHECK: %8 = "irdsl_test.mixed"(%c1_i32, %c1_i32) {in2 = 2 : i32, in4 = 2 : i32, operandSegmentSizes = array<i32: 1, 0, 1>} : (i32, i32) -> i32
            m1 = MixedOp(i32, ione, iattr, iattr, ione)
            # CHECK: %9 = "irdsl_test.mixed"(%c1_i32, %c1_i32, %c1_i32) {in2 = 2 : i32, in4 = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 1>} : (i32, i32, i32) -> i32
            m2 = MixedOp(i32, ione, iattr, iattr, ione, in3=ione)

        print(module)
        assert module.operation.verify()

        # CHECK: Value(%c1_i32
        print(c1.a)
        # CHECK: 2 : i32
        print(c1.x)
        # CHECK: Value(%c1_i32
        print(o1.a)
        # CHECK: None
        print(o1.b)
        # CHECK: Value(%c1_i32
        print(o2.b)
        # CHECK: 0
        print(o1.out1.result_number)
        # CHECK: None
        print(o1.out2)
        # CHECK: 0
        print(o2.out1.result_number)
        # CHECK: 1
        print(o2.out2.result_number)
        # CHECK: None
        print(o3.a)
        # CHECK: Value(%c1_i32
        print(o5.a)
        # CHECK: ['Value(%c1_i32 = arith.constant 1 : i32)', 'Value(%c1_i32 = arith.constant 1 : i32)']
        print([str(i) for i in v1.c])
        # CHECK: ['Value(%c1_i32 = arith.constant 1 : i32)']
        print([str(i) for i in v2.c])
        # CHECK: []
        print([str(i) for i in v3.c])
        # CHECK: 0 0
        print(len(v4.a), len(v4.b))
        # CHECK: 3 0
        print(len(v5.a), len(v5.b))
        # CHECK: 1 2
        print(len(v6.a), len(v6.b))
