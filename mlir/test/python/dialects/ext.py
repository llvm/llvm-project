# RUN: %PYTHON %s 2>&1 | FileCheck %s

from mlir.ir import *
from mlir.dialects import arith
from mlir.dialects.ext import *
from typing import Any, Optional, Sequence, TypeVar, Union
import sys


def run(f):
    print("\nTEST:", f.__name__)
    f()


# CHECK: TEST: testMyInt
@run
def testMyInt():
    class MyInt(Dialect, name="myint"):
        pass

    i32 = IntegerType[32]

    class ConstantOp(MyInt.Operation, name="constant"):
        value: IntegerAttr
        cst: Result[i32]

    class AddOp(MyInt.Operation, name="add"):
        lhs: Operand[i32]
        rhs: Operand[i32]
        res: Result[i32]

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
        MyInt.load()
        print(MyInt._mlir_module)

        # CHECK: ['constant', 'add']
        print([i._op_name for i in MyInt.operations])
        i32 = IntegerType.get_signless(32)

        module = Module.create()
        with InsertionPoint(module.body):
            two = ConstantOp(IntegerAttr.get(i32, 2))
            three = ConstantOp(IntegerAttr.get(i32, 3))
            add1 = AddOp(two, three)
            add2 = AddOp(add1, two)
            add3 = AddOp(add2, three)

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
        # CHECK: OpResult(%0
        print(two.cst)
        # CHECK: (self, /, lhs, rhs, *, loc=None, ip=None)
        print(AddOp.__init__.__signature__)
        # CHECK: (self, /, value, *, loc=None, ip=None)
        print(ConstantOp.__init__.__signature__)


# CHECK: TEST: testExtDialect
@run
def testExtDialect():
    class Test(Dialect, name="ext_test"):
        pass

    i32 = IntegerType[32]

    class ConstraintOp(Test.Operation, name="constraint"):
        a: Operand[i32 | IntegerType[64]]
        b: Operand[Any]
        # Here we use `F32Type[()]` instead of just `F32Type`
        # because of an existing issue in IRDL implementation
        # where `irdl.base` cannot exist in `irdl.any_of`.
        c: Operand[F32Type[()] | i32]
        d: Operand[Any]
        x: IntegerAttr
        y: FloatAttr

    class OptionalOp(Test.Operation, name="optional"):
        a: Operand[i32]
        b: Optional[Operand[i32]]
        out1: Result[i32]
        out2: Result[i32] | None
        out3: Result[i32]

    class Optional2Op(Test.Operation, name="optional2"):
        a: Optional[Operand[i32]]
        b: Optional[Result[i32]]

    class VariadicOp(Test.Operation, name="variadic"):
        a: Operand[i32]
        b: Optional[Operand[i32]]
        c: Sequence[Operand[i32]]
        out1: Sequence[Result[i32]]
        out2: Sequence[Result[i32]]
        out3: Optional[Result[i32]]
        out4: Result[i32]

    class Variadic2Op(Test.Operation, name="variadic2"):
        a: Sequence[Operand[i32]]
        b: Sequence[Result[i32]]

    class MixedOpBase(Test.Operation):
        out: Result[i32]
        in1: Operand[i32]

    class MixedOp(MixedOpBase, name="mixed"):
        in2: IntegerAttr
        in3: Optional[Operand[i32]]
        in4: IntegerAttr
        in5: Operand[i32]

    T = TypeVar("T")
    U = TypeVar("U", bound=IntegerType[32] | IntegerType[64])
    V = TypeVar("V", bound=Union[IntegerType[8], IntegerType[16]])

    class TypeVarOp(Test.Operation, name="type_var"):
        in1: Operand[T]
        in2: Operand[T]
        in3: Operand[U]
        in4: Operand[U | V]
        in5: Operand[V]

    # CHECK: irdl.dialect @ext_test {
    # CHECK:   irdl.operation @constraint {
    # CHECK:     %0 = irdl.is i32
    # CHECK:     %1 = irdl.is i64
    # CHECK:     %2 = irdl.any_of(%0, %1)
    # CHECK:     %3 = irdl.any
    # CHECK:     %4 = irdl.is f32
    # CHECK:     %5 = irdl.any_of(%4, %0)
    # CHECK:     %6 = irdl.any
    # CHECK:     irdl.operands(a: %2, b: %3, c: %5, d: %6)
    # CHECK:     %7 = irdl.base "#builtin.integer"
    # CHECK:     %8 = irdl.base "#builtin.float"
    # CHECK:     irdl.attributes {"x" = %7, "y" = %8}
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
    # CHECK:     %2 = irdl.base "#builtin.integer"
    # CHECK:     irdl.attributes {"in2" = %1, "in4" = %2}
    # CHECK:     irdl.results(out: %0)
    # CHECK:   }
    # CHECK:   irdl.operation @type_var {
    # CHECK:     %0 = irdl.any
    # CHECK:     %1 = irdl.is i32
    # CHECK:     %2 = irdl.is i64
    # CHECK:     %3 = irdl.any_of(%1, %2)
    # CHECK:     %4 = irdl.is i8
    # CHECK:     %5 = irdl.is i16
    # CHECK:     %6 = irdl.any_of(%4, %5)
    # CHECK:     %7 = irdl.any_of(%3, %6)
    # CHECK:     irdl.operands(in1: %0, in2: %0, in3: %3, in4: %7, in5: %6)
    # CHECK:   }
    # CHECK: }
    with Context(), Location.unknown():
        Test.load()
        print(Test._mlir_module)

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
        # CHECK: (self, /, in1, in2, in4, in5, *, in3=None, loc=None, ip=None)
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

            # CHECK: "ext_test.constraint"(%c1_i32, %c1_i32, %cst, %c1_i32) {x = 2 : i32, y = 2.300000e+00 : f32} : (i32, i32, f32, i32) -> ()
            c1 = ConstraintOp(ione, ione, fone, ione, iattr, fattr)
            # CHECK: "ext_test.constraint"(%c1_i32, %cst, %cst, %cst) {x = 2 : i32, y = 2.300000e+00 : f32} : (i32, f32, f32, f32) -> ()
            ConstraintOp(ione, fone, fone, fone, iattr, fattr)
            # CHECK: ext_test.constraint"(%c1_i32, %cst, %c1_i32, %cst) {x = 2 : i32, y = 2.300000e+00 : f32} : (i32, f32, i32, f32) -> ()
            ConstraintOp(ione, fone, ione, fone, iattr, fattr)

            # CHECK: %0:2 = "ext_test.optional"(%c1_i32) {operandSegmentSizes = array<i32: 1, 0>, resultSegmentSizes = array<i32: 1, 0, 1>} : (i32) -> (i32, i32)
            o1 = OptionalOp(i32, i32, ione)
            # CHECK: %1:3 = "ext_test.optional"(%c1_i32, %c1_i32) {operandSegmentSizes = array<i32: 1, 1>, resultSegmentSizes = array<i32: 1, 1, 1>} : (i32, i32) -> (i32, i32, i32)
            o2 = OptionalOp(i32, i32, ione, out2=i32, b=ione)
            # CHECK: ext_test.optional2"() {operandSegmentSizes = array<i32: 0>, resultSegmentSizes = array<i32: 0>} : () -> ()
            o3 = Optional2Op()
            # CHECK: %2 = "ext_test.optional2"() {operandSegmentSizes = array<i32: 0>, resultSegmentSizes = array<i32: 1>} : () -> i32
            o4 = Optional2Op(b=i32)
            # CHECK: "ext_test.optional2"(%c1_i32) {operandSegmentSizes = array<i32: 1>, resultSegmentSizes = array<i32: 0>} : (i32) -> ()
            o5 = Optional2Op(a=ione)
            # CHECK: %3 = "ext_test.optional2"(%c1_i32) {operandSegmentSizes = array<i32: 1>, resultSegmentSizes = array<i32: 1>} : (i32) -> i32
            o6 = Optional2Op(b=i32, a=ione)

            # CHECK: %4:4 = "ext_test.variadic"(%c1_i32, %c1_i32, %c1_i32) {operandSegmentSizes = array<i32: 1, 0, 2>, resultSegmentSizes = array<i32: 1, 2, 0, 1>} : (i32, i32, i32) -> (i32, i32, i32, i32)
            v1 = VariadicOp([i32], [i32, i32], i32, ione, [ione, ione])
            # CHECK: %5:5 = "ext_test.variadic"(%c1_i32, %c1_i32, %c1_i32) {operandSegmentSizes = array<i32: 1, 1, 1>, resultSegmentSizes = array<i32: 1, 2, 1, 1>} : (i32, i32, i32) -> (i32, i32, i32, i32, i32)
            v2 = VariadicOp([i32], [i32, i32], i32, ione, [ione], out3=i32, b=ione)
            # CHECK: %6:4 = "ext_test.variadic"(%c1_i32) {operandSegmentSizes = array<i32: 1, 0, 0>, resultSegmentSizes = array<i32: 2, 1, 0, 1>} : (i32) -> (i32, i32, i32, i32)
            v3 = VariadicOp([i32, i32], [i32], i32, ione, [])
            # CHECK: "ext_test.variadic2"() {operandSegmentSizes = array<i32: 0>, resultSegmentSizes = array<i32: 0>} : () -> ()
            v4 = Variadic2Op([], [])
            # CHECK: "ext_test.variadic2"(%c1_i32, %c1_i32, %c1_i32) {operandSegmentSizes = array<i32: 3>, resultSegmentSizes = array<i32: 0>} : (i32, i32, i32) -> ()
            v5 = Variadic2Op([], [ione, ione, ione])
            # CHECK: %7:2 = "ext_test.variadic2"(%c1_i32) {operandSegmentSizes = array<i32: 1>, resultSegmentSizes = array<i32: 2>} : (i32) -> (i32, i32)
            v6 = Variadic2Op([i32, i32], [ione])

            # CHECK: %8 = "ext_test.mixed"(%c1_i32, %c1_i32) {in2 = 2 : i32, in4 = 2 : i32, operandSegmentSizes = array<i32: 1, 0, 1>} : (i32, i32) -> i32
            m1 = MixedOp(ione, iattr, iattr, ione)
            # CHECK: %9 = "ext_test.mixed"(%c1_i32, %c1_i32, %c1_i32) {in2 = 2 : i32, in4 = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 1>} : (i32, i32, i32) -> i32
            m2 = MixedOp(ione, iattr, iattr, ione, in3=ione)

        print(module)
        assert module.operation.verify()

        # CHECK: OpResult(%c1_i32
        print(c1.a)
        # CHECK: 2 : i32
        print(c1.x)
        # CHECK: OpResult(%c1_i32
        print(o1.a)
        # CHECK: None
        print(o1.b)
        # CHECK: OpResult(%c1_i32
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
        # CHECK: OpResult(%c1_i32
        print(o5.a)
        # CHECK: ['OpResult(%c1_i32 = arith.constant 1 : i32)', 'OpResult(%c1_i32 = arith.constant 1 : i32)']
        print([str(i) for i in v1.c])
        # CHECK: ['OpResult(%c1_i32 = arith.constant 1 : i32)']
        print([str(i) for i in v2.c])
        # CHECK: []
        print([str(i) for i in v3.c])
        # CHECK: 0 0
        print(len(v4.a), len(v4.b))
        # CHECK: 3 0
        print(len(v5.a), len(v5.b))
        # CHECK: 1 2
        print(len(v6.a), len(v6.b))

        # cases to violate constraits
        module = Module.create()
        with InsertionPoint(module.body):
            try:
                c1 = ConstraintOp(ione, ione, fone, ione, iattr)
            except TypeError as e:
                # CHECK: missing a required argument: 'y'
                print(e)

            try:
                c2 = ConstraintOp(ione, ione, fone, ione, iattr, fattr, ione)
            except TypeError as e:
                # CHECK：too many positional arguments
                print(e)
