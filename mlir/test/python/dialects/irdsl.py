# RUN: %PYTHON %s 2>&1 | FileCheck %s

from mlir.ir import *
from mlir.dialects.irdl import dsl as irdsl
from mlir.dialects import arith
import sys


def run(f):
    print("\nTEST:", f.__name__, file=sys.stderr)
    with Context():
        f()


# CHECK: TEST: testMyInt
@run
def testMyInt():
    myint = irdsl.Dialect("myint")
    iattr = irdsl.BaseName("#builtin.integer")
    i32 = irdsl.IsType(IntegerType.get_signless(32))

    @myint.op("constant")
    class ConstantOp:
        value = irdsl.Attribute(iattr)
        cst = irdsl.Result(i32)

    @myint.op("add")
    class AddOp:
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
    print(myint._make_module())
    myint = myint.load()

    # CHECK: ['ConstantOp', 'constant', 'AddOp', 'add']
    print([i for i in myint.__dict__.keys()])

    i32 = IntegerType.get_signless(32)
    with Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            two = myint.constant(i32, IntegerAttr.get(i32, 2))
            three = myint.constant(i32, IntegerAttr.get(i32, 3))
            add1 = myint.add(i32, two, three)
            add2 = myint.add(i32, add1, two)
            add3 = myint.add(i32, add2, three)

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
    # CHECK: (res, lhs, rhs, *, loc=None, ip=None)
    print(myint.add.__signature__)
    # CHECK: (cst, value, *, loc=None, ip=None)
    print(myint.constant.__signature__)


@run
def testIRDSL():
    test = irdsl.Dialect("irdsl_test")
    i32 = irdsl.IsType(IntegerType.get_signless(32))
    i64 = irdsl.IsType(IntegerType.get_signless(64))
    i32or64 = i32 | i64
    any = irdsl.Any()
    f32 = irdsl.IsType(F32Type.get())
    iattr = irdsl.BaseName("#builtin.integer")
    fattr = irdsl.BaseName("#builtin.float")

    @test.op("constraint")
    class ConstraintOp:
        a = irdsl.Operand(i32or64)
        b = irdsl.Operand(any)
        c = irdsl.Operand(f32 | i32)
        d = irdsl.Operand(any)
        x = irdsl.Attribute(iattr)
        y = irdsl.Attribute(fattr)

    @test.op("optional")
    class OptionalOp:
        a = irdsl.Operand(i32)
        b = irdsl.Operand(i32, irdsl.Variadicity.optional)
        out1 = irdsl.Result(i32)
        out2 = irdsl.Result(i32, irdsl.Variadicity.optional)
        out3 = irdsl.Result(i32)

    @test.op("optional2")
    class Optional2Op:
        a = irdsl.Operand(i32, irdsl.Variadicity.optional)
        b = irdsl.Result(i32, irdsl.Variadicity.optional)

    @test.op("variadic")
    class VariadicOp:
        a = irdsl.Operand(i32)
        b = irdsl.Operand(i32, irdsl.Variadicity.optional)
        c = irdsl.Operand(i32, irdsl.Variadicity.variadic)
        out1 = irdsl.Result(i32, irdsl.Variadicity.variadic)
        out2 = irdsl.Result(i32, irdsl.Variadicity.variadic)
        out3 = irdsl.Result(i32, irdsl.Variadicity.optional)
        out4 = irdsl.Result(i32)

    @test.op("variadic2")
    class Variadic2Op:
        a = irdsl.Operand(i32, irdsl.Variadicity.variadic)
        b = irdsl.Result(i32, irdsl.Variadicity.variadic)

    @test.op("mixed")
    class MixedOp:
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
    print(test._make_module())
    test = test.load()

    # CHECK: (a, b, c, d, x, y, *, loc=None, ip=None)
    print(test.constraint.__signature__)
    # CHECK: (out1, out3, a, *, out2=None, b=None, loc=None, ip=None)
    print(test.optional.__signature__)
    # CHECK: (*, b=None, a=None, loc=None, ip=None)
    print(test.optional2.__signature__)
    # CHECK: (out1, out2, out4, a, c, *, out3=None, b=None, loc=None, ip=None)
    print(test.variadic.__signature__)
    # CHECK: (b, a, *, loc=None, ip=None)
    print(test.variadic2.__signature__)
    # CHECK: (out, in1, in2, in4, in5, *, in3=None, loc=None, ip=None)
    print(test.mixed.__signature__)

    # CHECK: None None
    print(
        test.ConstraintOp._ODS_OPERAND_SEGMENTS, test.ConstraintOp._ODS_RESULT_SEGMENTS
    )
    # CHECK: [1, 0] [1, 0, 1]
    print(test.OptionalOp._ODS_OPERAND_SEGMENTS, test.OptionalOp._ODS_RESULT_SEGMENTS)
    # CHECK: [0] [0]
    print(test.Optional2Op._ODS_OPERAND_SEGMENTS, test.Optional2Op._ODS_RESULT_SEGMENTS)
    # CHECK: [1, 0, -1] [-1, -1, 0, 1]
    print(test.VariadicOp._ODS_OPERAND_SEGMENTS, test.VariadicOp._ODS_RESULT_SEGMENTS)
    # CHECK: [-1] [-1]
    print(test.Variadic2Op._ODS_OPERAND_SEGMENTS, test.Variadic2Op._ODS_RESULT_SEGMENTS)

    i32 = IntegerType.get_signless(32)
    i64 = IntegerType.get_signless(64)
    f32 = F32Type.get()

    with Location.unknown():
        iattr = IntegerAttr.get(i32, 2)
        fattr = FloatAttr.get_f32(2.3)

        module = Module.create()
        with InsertionPoint(module.body):
            ione = arith.constant(i32, 1)
            fone = arith.constant(f32, 1.2)

            # CHECK: "irdsl_test.constraint"(%c1_i32, %c1_i32, %cst, %c1_i32) {x = 2 : i32, y = 2.300000e+00 : f32} : (i32, i32, f32, i32) -> ()
            c1 = test.constraint(ione, ione, fone, ione, iattr, fattr)
            # CHECK: "irdsl_test.constraint"(%c1_i32, %cst, %cst, %cst) {x = 2 : i32, y = 2.300000e+00 : f32} : (i32, f32, f32, f32) -> ()
            test.constraint(ione, fone, fone, fone, iattr, fattr)
            # CHECK: irdsl_test.constraint"(%c1_i32, %cst, %c1_i32, %cst) {x = 2 : i32, y = 2.300000e+00 : f32} : (i32, f32, i32, f32) -> ()
            test.constraint(ione, fone, ione, fone, iattr, fattr)

            # CHECK: %0:2 = "irdsl_test.optional"(%c1_i32) {operandSegmentSizes = array<i32: 1, 0>, resultSegmentSizes = array<i32: 1, 0, 1>} : (i32) -> (i32, i32)
            o1 = test.optional(i32, i32, ione)
            # CHECK: %1:3 = "irdsl_test.optional"(%c1_i32, %c1_i32) {operandSegmentSizes = array<i32: 1, 1>, resultSegmentSizes = array<i32: 1, 1, 1>} : (i32, i32) -> (i32, i32, i32)
            o2 = test.optional(i32, i32, ione, out2=i32, b=ione)
            # CHECK: irdsl_test.optional2"() {operandSegmentSizes = array<i32: 0>, resultSegmentSizes = array<i32: 0>} : () -> ()
            o3 = test.optional2()
            # CHECK: %2 = "irdsl_test.optional2"() {operandSegmentSizes = array<i32: 0>, resultSegmentSizes = array<i32: 1>} : () -> i32
            o4 = test.optional2(b=i32)
            # CHECK: "irdsl_test.optional2"(%c1_i32) {operandSegmentSizes = array<i32: 1>, resultSegmentSizes = array<i32: 0>} : (i32) -> ()
            o5 = test.optional2(a=ione)
            # CHECK: %3 = "irdsl_test.optional2"(%c1_i32) {operandSegmentSizes = array<i32: 1>, resultSegmentSizes = array<i32: 1>} : (i32) -> i32
            o6 = test.optional2(b=i32, a=ione)

            # CHECK: %4:4 = "irdsl_test.variadic"(%c1_i32, %c1_i32, %c1_i32) {operandSegmentSizes = array<i32: 1, 0, 2>, resultSegmentSizes = array<i32: 1, 2, 0, 1>} : (i32, i32, i32) -> (i32, i32, i32, i32)
            v1 = test.variadic([i32], [i32, i32], i32, ione, [ione, ione])
            # CHECK: %5:5 = "irdsl_test.variadic"(%c1_i32, %c1_i32, %c1_i32) {operandSegmentSizes = array<i32: 1, 1, 1>, resultSegmentSizes = array<i32: 1, 2, 1, 1>} : (i32, i32, i32) -> (i32, i32, i32, i32, i32)
            v2 = test.variadic([i32], [i32, i32], i32, ione, [ione], out3=i32, b=ione)
            # CHECK: %6:4 = "irdsl_test.variadic"(%c1_i32) {operandSegmentSizes = array<i32: 1, 0, 0>, resultSegmentSizes = array<i32: 2, 1, 0, 1>} : (i32) -> (i32, i32, i32, i32)
            v3 = test.variadic([i32, i32], [i32], i32, ione, [])
            # CHECK: "irdsl_test.variadic2"() {operandSegmentSizes = array<i32: 0>, resultSegmentSizes = array<i32: 0>} : () -> ()
            v4 = test.variadic2([], [])
            # CHECK: "irdsl_test.variadic2"(%c1_i32, %c1_i32, %c1_i32) {operandSegmentSizes = array<i32: 3>, resultSegmentSizes = array<i32: 0>} : (i32, i32, i32) -> ()
            v5 = test.variadic2([], [ione, ione, ione])
            # CHECK: %7:2 = "irdsl_test.variadic2"(%c1_i32) {operandSegmentSizes = array<i32: 1>, resultSegmentSizes = array<i32: 2>} : (i32) -> (i32, i32)
            v6 = test.variadic2([i32, i32], [ione])

            # CHECK: %8 = "irdsl_test.mixed"(%c1_i32, %c1_i32) {in2 = 2 : i32, in4 = 2 : i32, operandSegmentSizes = array<i32: 1, 0, 1>} : (i32, i32) -> i32
            m1 = test.mixed(i32, ione, iattr, iattr, ione)
            # CHECK: %9 = "irdsl_test.mixed"(%c1_i32, %c1_i32, %c1_i32) {in2 = 2 : i32, in4 = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 1>} : (i32, i32, i32) -> i32
            m2 = test.mixed(i32, ione, iattr, iattr, ione, in3=ione)

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
