# RUN: %PYTHON %s 2>&1 | FileCheck %s

from mlir.ir import *
from mlir.dialects import arith
from mlir.dialects.ext import *
from mlir import ir
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
        cst: Result[i32] = result(infer_type=True)

    class AddOp(Operation, dialect=MyInt, name="add"):
        lhs: Operand[i32]
        rhs: Operand[i32]
        res: Result[i32] = result(infer_type=True)

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
    with Context():
        MyInt.load()
        print(MyInt._mlir_module)

        # CHECK: ['constant', 'add']
        print([i._op_name for i in MyInt.operations])
        i32 = IntegerType.get_signless(32)

        with Location.unknown():
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
        # CHECK: (0, True)
        print(add1._ODS_REGIONS)
        # CHECK: %0 = "myint.constant"() {value = 2 : i32} : () -> i32
        print(add1.lhs.owner)
        # CHECK: %1 = "myint.constant"() {value = 3 : i32} : () -> i32
        print(add1.rhs.owner)
        # CHECK: 2 : i32
        print(two.value)
        # CHECK: OpResult(%0
        print(two.cst)
        # CHECK: (self, /, lhs, rhs, *, res=None, loc=None, ip=None)
        print(AddOp.__init__.__signature__)
        # CHECK: (self, /, value, *, cst=None, loc=None, ip=None)
        print(ConstantOp.__init__.__signature__)

        # CHECK: True
        print(issubclass(AddOp.Adaptor, OpAdaptor))
        adaptor1 = AddOp.Adaptor(list(add1.operands), add1)
        # CHECK: myint.add
        print(adaptor1.OPERATION_NAME)
        # CHECK: OpResult(%0 = "myint.constant"() {value = 2 : i32} : () -> i32)
        print(adaptor1.lhs)
        # CHECK: OpResult(%1 = "myint.constant"() {value = 3 : i32} : () -> i32)
        print(adaptor1.rhs)


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
        out1: Result[i32]
        out3: Result[i32]
        a: Operand[i32]
        out2: Result[i32] | None = None
        b: Optional[Operand[i32]] = None

    class Optional2Op(Test.Operation, name="optional2"):
        b: Optional[Result[i32]] = None
        a: Optional[Operand[i32]] = None

    class VariadicOp(Test.Operation, name="variadic"):
        out1: Sequence[Result[i32]]
        out2: Sequence[Result[i32]]
        out4: Result[i32]
        a: Operand[i32]
        c: Sequence[Operand[i32]]
        out3: Optional[Result[i32]] = None
        b: Optional[Operand[i32]] = None

    class Variadic2Op(Test.Operation, name="variadic2"):
        b: Sequence[Result[i32]]
        a: Sequence[Operand[i32]]

    class MixedOpBase(Test.Operation):
        out: Result[i32]
        in1: Operand[i32]

    class MixedOp(MixedOpBase, name="mixed"):
        in2: IntegerAttr
        in4: IntegerAttr
        in5: Operand[i32]
        in3: Optional[Operand[i32]] = None

    T = TypeVar("T")
    U = TypeVar("U", bound=IntegerType[32] | IntegerType[64])
    V = TypeVar("V", bound=Union[IntegerType[8], IntegerType[16]])

    class TypeVarOp(Test.Operation, name="type_var"):
        in1: Operand[T]
        in2: Operand[T]
        in3: Operand[U]
        in4: Operand[U | V]
        in5: Operand[V]

    class OptionalButNotKeywordOp(Test.Operation, name="optional_but_not_keyword"):
        a: Operand[i32]
        b: Optional[Operand[i32]]
        c: Operand[i32]

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
    # CHECK:     irdl.results(out1: %0, out3: %0, out2: optional %0)
    # CHECK:   }
    # CHECK:   irdl.operation @optional2 {
    # CHECK:     %0 = irdl.is i32
    # CHECK:     irdl.operands(a: optional %0)
    # CHECK:     irdl.results(b: optional %0)
    # CHECK:   }
    # CHECK:   irdl.operation @variadic {
    # CHECK:     %0 = irdl.is i32
    # CHECK:     irdl.operands(a: %0, c: variadic %0, b: optional %0)
    # CHECK:     irdl.results(out1: variadic %0, out2: variadic %0, out4: %0, out3: optional %0)
    # CHECK:   }
    # CHECK:   irdl.operation @variadic2 {
    # CHECK:     %0 = irdl.is i32
    # CHECK:     irdl.operands(a: variadic %0)
    # CHECK:     irdl.results(b: variadic %0)
    # CHECK:   }
    # CHECK:   irdl.operation @mixed {
    # CHECK:     %0 = irdl.is i32
    # CHECK:     irdl.operands(in1: %0, in5: %0, in3: optional %0)
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
        # CHECK: (self, /, out, in1, in2, in4, in5, *, in3=None, loc=None, ip=None)
        print(MixedOp.__init__.__signature__)
        # CHECK: (self, /, in1, in2, in3, in4, in5, *, loc=None, ip=None)
        print(TypeVarOp.__init__.__signature__)
        # CHECK: (self, /, a, b, c, *, loc=None, ip=None)
        print(OptionalButNotKeywordOp.__init__.__signature__)

        # CHECK: None None
        print(ConstraintOp._ODS_OPERAND_SEGMENTS, ConstraintOp._ODS_RESULT_SEGMENTS)
        # CHECK: [1, 0] [1, 1, 0]
        print(OptionalOp._ODS_OPERAND_SEGMENTS, OptionalOp._ODS_RESULT_SEGMENTS)
        # CHECK: [0] [0]
        print(Optional2Op._ODS_OPERAND_SEGMENTS, Optional2Op._ODS_RESULT_SEGMENTS)
        # CHECK: [1, -1, 0] [-1, -1, 1, 0]
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

            # CHECK: %0:2 = "ext_test.optional"(%c1_i32) {operandSegmentSizes = array<i32: 1, 0>, resultSegmentSizes = array<i32: 1, 1, 0>} : (i32) -> (i32, i32)
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

            # CHECK: %4:4 = "ext_test.variadic"(%c1_i32, %c1_i32, %c1_i32) {operandSegmentSizes = array<i32: 1, 2, 0>, resultSegmentSizes = array<i32: 1, 2, 1, 0>} : (i32, i32, i32) -> (i32, i32, i32, i32)
            v1 = VariadicOp([i32], [i32, i32], i32, ione, [ione, ione])
            # CHECK: %5:5 = "ext_test.variadic"(%c1_i32, %c1_i32, %c1_i32) {operandSegmentSizes = array<i32: 1, 1, 1>, resultSegmentSizes = array<i32: 1, 2, 1, 1>} : (i32, i32, i32) -> (i32, i32, i32, i32, i32)
            v2 = VariadicOp([i32], [i32, i32], i32, ione, [ione], out3=i32, b=ione)
            # CHECK: %6:4 = "ext_test.variadic"(%c1_i32) {operandSegmentSizes = array<i32: 1, 0, 0>, resultSegmentSizes = array<i32: 2, 1, 1, 0>} : (i32) -> (i32, i32, i32, i32)
            v3 = VariadicOp([i32, i32], [i32], i32, ione, [])
            # CHECK: "ext_test.variadic2"() {operandSegmentSizes = array<i32: 0>, resultSegmentSizes = array<i32: 0>} : () -> ()
            v4 = Variadic2Op([], [])
            # CHECK: "ext_test.variadic2"(%c1_i32, %c1_i32, %c1_i32) {operandSegmentSizes = array<i32: 3>, resultSegmentSizes = array<i32: 0>} : (i32, i32, i32) -> ()
            v5 = Variadic2Op([], [ione, ione, ione])
            # CHECK: %7:2 = "ext_test.variadic2"(%c1_i32) {operandSegmentSizes = array<i32: 1>, resultSegmentSizes = array<i32: 2>} : (i32) -> (i32, i32)
            v6 = Variadic2Op([i32, i32], [ione])

            # CHECK: %8 = "ext_test.mixed"(%c1_i32, %c1_i32) {in2 = 2 : i32, in4 = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 0>} : (i32, i32) -> i32
            m1 = MixedOp(i32, ione, iattr, iattr, ione)
            # CHECK: %9 = "ext_test.mixed"(%c1_i32, %c1_i32, %c1_i32) {in2 = 2 : i32, in4 = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 1>} : (i32, i32, i32) -> i32
            m2 = MixedOp(i32, ione, iattr, iattr, ione, in3=ione)

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
        # CHECK: 2
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


# CHECK: TEST: testExtDialectWithRegion
@run
def testExtDialectWithRegion():
    class ParentIsIfTrait(DynamicOpTrait):
        @staticmethod
        def verify_invariants(op) -> bool:
            if not isinstance(op.parent.opview, IfOp):
                op.location.emit_error(
                    f"{op.name} should be put inside {IfOp.OPERATION_NAME}"
                )
                return False
            return True

    class TestRegion(Dialect, name="ext_region"):
        pass

    class IfOp(TestRegion.Operation, name="if"):
        result: Result[Any]
        cond: Operand[IntegerType[1]]
        then: Region
        else_: Region

    class YieldOp(
        TestRegion.Operation, name="yield", traits=[IsTerminatorTrait, ParentIsIfTrait]
    ):
        value: Operand[Any]

        def verify_invariants(self) -> bool:
            if self.parent.results[0].type != self.value.type:
                self.location.emit_error(
                    "result type mismatch between YieldOp and its parent IfOp"
                )
                return False
            return True

    class NoTermOp(TestRegion.Operation, name="no_term", traits=[NoTerminatorTrait]):
        body: Region

    with Context(), Location.unknown():
        TestRegion.load()
        # CHECK: irdl.dialect @ext_region {
        # CHECK:   irdl.operation @if {
        # CHECK:     %0 = irdl.is i1
        # CHECK:     irdl.operands(cond: %0)
        # CHECK:     %1 = irdl.any
        # CHECK:     irdl.results(result: %1)
        # CHECK:     %2 = irdl.region
        # CHECK:     %3 = irdl.region
        # CHECK:     irdl.regions(then: %2, else_: %3)
        # CHECK:   }
        # CHECK:   irdl.operation @yield {
        # CHECK:     %0 = irdl.any
        # CHECK:     irdl.operands(value: %0)
        # CHECK:   }
        # CHECK:   irdl.operation @no_term {
        # CHECK:     %0 = irdl.region
        # CHECK:     irdl.regions(body: %0)
        # CHECK:   }
        # CHECK: }
        print(TestRegion._mlir_module)

        # CHECK: (self, /, result, cond, *, loc=None, ip=None)
        print(IfOp.__init__.__signature__)

        # CHECK: None None
        print(IfOp._ODS_OPERAND_SEGMENTS, IfOp._ODS_RESULT_SEGMENTS)
        # CHECK: (2, True)
        print(IfOp._ODS_REGIONS)

        module = Module.create()
        with InsertionPoint(module.body):
            i1 = IntegerType.get_signless(1)
            i32 = IntegerType.get_signless(32)
            cond = arith.constant(i1, 1)

            if_ = IfOp(i32, cond)
            if_.then.blocks.append()
            if_.else_.blocks.append()

            with InsertionPoint(if_.then.blocks[0]):
                v = arith.constant(i32, 2)
                yield_ = YieldOp(v)

            with InsertionPoint(if_.else_.blocks[0]):
                v = arith.constant(i32, 3)
                YieldOp(v)

            nt = NoTermOp()
            nt.body.blocks.append()

            with InsertionPoint(nt.body.blocks[0]):
                arith.constant(i32, 4)
                # No terminator here

        assert module.operation.verify()
        # CHECK: module {
        # CHECK:     %true = arith.constant true
        # CHECK:     %0 = "ext_region.if"(%true) ({
        # CHECK:         %c2_i32 = arith.constant 2 : i32
        # CHECK:         "ext_region.yield"(%c2_i32) : (i32) -> ()
        # CHECK:     }, {
        # CHECK:         %c3_i32 = arith.constant 3 : i32
        # CHECK:         "ext_region.yield"(%c3_i32) : (i32) -> ()
        # CHECK:     }) : (i1) -> i32
        # CHECK:     "ext_region.no_term"() ({
        # CHECK:       %c4_i32 = arith.constant 4 : i32
        # CHECK:     }) : () -> ()
        # CHECK: }
        print(module)

        # CHECK: True
        print(yield_.has_trait(IsTerminatorTrait))
        # CHECK: False
        print(yield_.has_trait(NoTerminatorTrait))
        # CHECK: True
        print(yield_.has_trait(ParentIsIfTrait))
        # CHECK: False
        print(nt.operation.has_trait(IsTerminatorTrait))
        # CHECK: True
        print(nt.operation.has_trait(NoTerminatorTrait))
        # CHECK: False
        print(nt.operation.has_trait(ParentIsIfTrait))
        # CHECK: False
        print(NoTermOp.has_trait(IsTerminatorTrait))
        # CHECK: True
        print(NoTermOp.has_trait(NoTerminatorTrait))
        # CHECK: False
        print(NoTermOp.has_trait(ParentIsIfTrait))

        # CHECK: %c2_i32 = arith.constant 2 : i32
        print(if_.then.blocks[0])
        # CHECK: %c3_i32 = arith.constant 3 : i32
        print(if_.else_.blocks[0])

        # CHECK-LABEL: Testing violation cases
        print("Testing violation cases:")

        module = Module.create()
        with InsertionPoint(module.body):
            i1 = IntegerType.get_signless(1)
            i32 = IntegerType.get_signless(32)
            cond = arith.constant(i1, 1)

            if_ = IfOp(i32, cond)
            if_.then.blocks.append()
            if_.else_.blocks.append()

            with InsertionPoint(if_.then.blocks[0]):
                v = arith.constant(i32, 2)

            with InsertionPoint(if_.else_.blocks[0]):
                v = arith.constant(i32, 3)

        try:
            module.operation.verify()
        except Exception as e:
            # CHECK: Verification failed:
            # CHECK: block with no terminator
            print(e)

        module = Module.create()
        with InsertionPoint(module.body):
            v = arith.constant(i32, 2)
            YieldOp(v)

        try:
            module.operation.verify()
        except Exception as e:
            # CHECK: Verification failed:
            # CHECK: ext_region.yield should be put inside ext_region.if
            print(e)

        module = Module.create()
        with InsertionPoint(module.body):
            i1 = IntegerType.get_signless(1)
            i32 = IntegerType.get_signless(32)
            cond = arith.constant(i1, 1)

            if_ = IfOp(i1, cond)
            if_.then.blocks.append()
            if_.else_.blocks.append()

            with InsertionPoint(if_.then.blocks[0]):
                v = arith.constant(i32, 2)
                YieldOp(v)

            with InsertionPoint(if_.else_.blocks[0]):
                v = arith.constant(i32, 3)
                YieldOp(v)

        try:
            module.operation.verify()
        except Exception as e:
            # CHECK: Verification failed:
            # CHECK: result type mismatch
            print(e)


# CHECK: TEST: testExtDialectWithType
@run
def testExtDialectWithType():
    class TestType(Dialect, name="ext_type"):
        pass

    class Array(TestType.Type, name="array"):
        elem_type: IntegerType[32] | IntegerType[64]
        length: IntegerAttr

    class MakeArrayOp(TestType.Operation, name="make_array"):
        arr: Result[Array]

    class MakeArray3Op(TestType.Operation, name="make_array3"):
        arr: Result[Array[IntegerType[32], IntegerAttr[IntegerType[32], 3]]] = result(
            infer_type=True
        )

    with Context(), Location.unknown():
        TestType.load()
        # CHECK: irdl.dialect @ext_type {
        # CHECK:   irdl.type @array {
        # CHECK:     %0 = irdl.is i32
        # CHECK:     %1 = irdl.is i64
        # CHECK:     %2 = irdl.any_of(%0, %1)
        # CHECK:     %3 = irdl.base "#builtin.integer"
        # CHECK:     irdl.parameters(elem_type: %2, length: %3)
        # CHECK:   }
        # CHECK:   irdl.operation @make_array {
        # CHECK:     %0 = irdl.base @ext_type::@array
        # CHECK:     irdl.results(arr: %0)
        # CHECK:   }
        # CHECK:   irdl.operation @make_array3 {
        # CHECK:     %0 = irdl.is i32
        # CHECK:     %1 = irdl.is 3 : i32
        # CHECK:     %2 = irdl.parametric @ext_type::@array<%0, %1>
        # CHECK:     irdl.results(arr: %2)
        # CHECK:   }
        # CHECK: }
        print(TestType._mlir_module)

        # CHECK: ext_type.array
        print(Array.type_name)

        i32 = IntegerType.get_signless(32)
        i64 = IntegerType.get_signless(64)
        a4 = Array.get(i32, IntegerAttr.get(i32, 4))
        a6 = Array.get(i64, IntegerAttr.get(i32, 6))
        # CHECK: !ext_type.array<i32, 4 : i32>
        print(a4)
        # CHECK: !ext_type.array<i64, 6 : i32>
        print(a6)

        # CHECK: i32
        print(a4.elem_type)
        # CHECK: 4 : i32
        print(a4.length)
        # CHECK: i64
        print(a6.elem_type)
        # CHECK: 6 : i32
        print(a6.length)

        # CHECK: <locals>.Array
        print(type(Type(a4).maybe_downcast()))

        module = Module.create()
        with InsertionPoint(module.body):
            MakeArrayOp(a4)
            MakeArrayOp(a6)
            MakeArray3Op()

        # CHECK: %0 = "ext_type.make_array"() : () -> !ext_type.array<i32, 4 : i32>
        # CHECK: %1 = "ext_type.make_array"() : () -> !ext_type.array<i64, 6 : i32>
        # CHECK: %2 = "ext_type.make_array3"() : () -> !ext_type.array<i32, 3 : i32>
        assert module.operation.verify()
        print(module)


# CHECK: TEST: testExtDialectWithAttr
@run
def testExtDialectWithAttr():
    class TestAttr(Dialect, name="ext_attr"):
        pass

    class IntPair(TestAttr.Attribute, name="pair"):
        first: IntegerAttr
        second: IntegerAttr

    class StrPair(TestAttr.Attribute, name="str_pair"):
        first: StringAttr
        second: StringAttr

    class Op1(TestAttr.Operation, name="op1"):
        pair: IntPair

    class Op2(TestAttr.Operation, name="op2"):
        pair: StrPair
        pair2: StrPair[StringAttr["a"], StringAttr["b"]]

    with Context(), Location.unknown():
        TestAttr.load()
        # CHECK: irdl.dialect @ext_attr {
        # CHECK:   irdl.attribute @pair {
        # CHECK:     %0 = irdl.base "#builtin.integer"
        # CHECK:     %1 = irdl.base "#builtin.integer"
        # CHECK:     irdl.parameters(first: %0, second: %1)
        # CHECK:   }
        # CHECK:   irdl.attribute @str_pair {
        # CHECK:     %0 = irdl.base "#builtin.string"
        # CHECK:     %1 = irdl.base "#builtin.string"
        # CHECK:     irdl.parameters(first: %0, second: %1)
        # CHECK:   }
        # CHECK:   irdl.operation @op1 {
        # CHECK:     %0 = irdl.base @ext_attr::@pair
        # CHECK:     irdl.attributes {"pair" = %0}
        # CHECK:   }
        # CHECK:   irdl.operation @op2 {
        # CHECK:     %0 = irdl.base @ext_attr::@str_pair
        # CHECK:     %1 = irdl.is "a"
        # CHECK:     %2 = irdl.is "b"
        # CHECK:     %3 = irdl.parametric @ext_attr::@str_pair<%1, %2>
        # CHECK:     irdl.attributes {"pair" = %0, "pair2" = %3}
        # CHECK:   }
        # CHECK: }
        print(TestAttr._mlir_module)

        # CHECK: ext_attr.pair
        print(IntPair.attr_name)

        # CHECK: ext_attr.str_pair
        print(StrPair.attr_name)

        ip = IntPair.get(
            IntegerAttr.get(IntegerType.get_signless(32), 1),
            IntegerAttr.get(IntegerType.get_signless(32), 2),
        )
        sp = StrPair.get(StringAttr.get("hello"), StringAttr.get("world"))
        # CHECK: #ext_attr.pair<1 : i32, 2 : i32>
        print(ip)
        # CHECK: #ext_attr.str_pair<"hello", "world">
        print(sp)

        # CHECK: "hello"
        print(sp.first)
        # CHECK: "world"
        print(sp.second)

        sp2 = ir.Attribute(sp).maybe_downcast()
        # CHECK: <locals>.StrPair
        print(type(sp2))
        # CHECK: #ext_attr.str_pair<"hello", "world">
        print(str(sp2))

        module = Module.create()
        with InsertionPoint(module.body):
            Op1(ip)
            p2 = StrPair.get(StringAttr.get("a"), StringAttr.get("b"))
            Op2(sp, p2)

        assert module.operation.verify()

        # CHECK: "ext_attr.op1"() {pair = #ext_attr.pair<1 : i32, 2 : i32>} : () -> ()
        # CHECK: "ext_attr.op2"() {pair = #ext_attr.str_pair<"hello", "world">, pair2 = #ext_attr.str_pair<"a", "b">} : () -> ()
        print(module)


# CHECK: TEST: testExtDialectWithInvalidOp
@run
def testExtDialectWithInvalidOp():
    class TestInvalid(Dialect, name="ext_invalid"):
        pass

    try:

        class InferTypeBeforePositionalOp(
            TestInvalid.Operation, name="infer_before_pos"
        ):
            res: Result[IntegerType[32]] = result(infer_type=True)
            a: Operand[IntegerType[32]]

    except ValueError as e:
        # CHECK: wrong parameter order
        print(e)

    try:

        class AssignNoneOnNonOptionalOp(
            TestInvalid.Operation, name="assign_none_on_non_optional"
        ):
            a: Operand[IntegerType[32]] = None

    except ValueError as e:
        # CHECK: only optional operand can be set to None
        print(e)

    try:

        class AssignNoneOnnAttributeOp(
            TestInvalid.Operation, name="assign_none_on_attribute"
        ):
            a: IntegerAttr = None

    except ValueError as e:
        # CHECK: only optional attribute can be set to None
        print(e)

    try:

        class CannotInferTypeOp(TestInvalid.Operation, name="cannot_infer_type"):
            a: Result[IntegerType] = result(infer_type=True)

    except TypeError as e:
        # CHECK: unsupported type for inferring
        print(e)

    try:

        class WrongFieldSpecifierOp(
            TestInvalid.Operation, name="wrong_field_specifier"
        ):
            a: Result[IntegerType] = operand()

    except TypeError as e:
        # CHECK: only `result` field specifier can be used for result fields
        print(e)

    try:

        class WrongFieldSpecifierOp2(
            TestInvalid.Operation, name="wrong_field_specifier2"
        ):
            a: IntegerAttr = operand()

    except TypeError as e:
        # CHECK: only `attribute` field specifier can be used for attribute fields
        print(e)


# CHECK: TEST: testExtDialectWithAttrInOp
@run
def testExtDialectWithAttrInOp():
    class TestAttrInOp(Dialect, name="ext_attr_in_op"):
        pass

    class OpWithAttr(TestAttrInOp.Operation, name="op_with_attr"):
        a: IntegerAttr | StringAttr
        b: IntegerType[32] | IntegerType[64]

    with Context(), Location.unknown():
        TestAttrInOp.load()
        # CHECK: irdl.dialect @ext_attr_in_op {
        # CHECK:   irdl.operation @op_with_attr {
        # CHECK:     %0 = irdl.base "#builtin.integer"
        # CHECK:     %1 = irdl.base "#builtin.string"
        # CHECK:     %2 = irdl.any_of(%0, %1)
        # CHECK:     %3 = irdl.is i32
        # CHECK:     %4 = irdl.is i64
        # CHECK:     %5 = irdl.any_of(%3, %4)
        # CHECK:     irdl.attributes {"a" = %2, "b" = %5}
        # CHECK:   }
        # CHECK: }
        print(TestAttrInOp._mlir_module)

        i32 = IntegerType.get_signless(32)
        i64 = IntegerType.get_signless(64)
        iattr = IntegerAttr.get(i32, 42)
        sattr = StringAttr.get("hello")

        module = Module.create()
        with InsertionPoint(module.body):
            OpWithAttr(iattr, TypeAttr.get(i32))
            OpWithAttr(sattr, TypeAttr.get(i64))

        assert module.operation.verify()
        # CHECK: "ext_attr_in_op.op_with_attr"() {a = 42 : i32, b = i32} : () -> ()
        # CHECK: "ext_attr_in_op.op_with_attr"() {a = "hello", b = i64} : () -> ()
        print(module)


@run
def testExtDialectFieldSpecifiers():
    class TestFieldSpecifiers(Dialect, name="ext_field_specifiers"):
        pass

    class OperandSpecifierOp(TestFieldSpecifiers.Operation, name="operand_specifier"):
        a: Operand[IntegerType[32]] = operand()
        b: Optional[Operand[IntegerType[32]]] = None
        c: Operand[IntegerType[32]] = operand(kw_only=True)

    class ResultSpecifierOp(TestFieldSpecifiers.Operation, name="result_specifier"):
        a: Result[IntegerType[32]] = result()
        b: Result[IntegerType[16]] = result(infer_type=True)
        c: Result[IntegerType] = result(
            default_factory=lambda: IntegerType.get_signless(8)
        )
        d: Sequence[Result[IntegerType]] = result(default_factory=list)
        e: Result[IntegerType[32]] = result(kw_only=True)

    class AttributeSpecifierOp(
        TestFieldSpecifiers.Operation, name="attribute_specifier"
    ):
        a: IntegerAttr = attribute()
        b: IntegerAttr = attribute(
            default_factory=lambda: IntegerAttr.get(IntegerType.get_signless(32), 42)
        )
        c: StringAttr["a"] | StringAttr["b"] = attribute(
            default_factory=lambda: StringAttr.get("a"), kw_only=True
        )
        d: IntegerAttr = attribute(kw_only=True)

    with Context(), Location.unknown():
        TestFieldSpecifiers.load()

        # CHECK: (self, /, a, *, b=None, c, loc=None, ip=None)
        print(OperandSpecifierOp.__init__.__signature__)
        # CHECK: (self, /, a, *, b=None, c=None, d=None, e, loc=None, ip=None)
        print(ResultSpecifierOp.__init__.__signature__)
        # CHECK: (self, /, a, *, b=None, c=None, d, loc=None, ip=None)
        print(AttributeSpecifierOp.__init__.__signature__)

        module = Module.create()
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            one = arith.constant(i32, 1)
            OperandSpecifierOp(one, c=one)
            ResultSpecifierOp(i32, e=i32)
            AttributeSpecifierOp(IntegerAttr.get(i32, 43), d=IntegerAttr.get(i32, 100))

        assert module.operation.verify()
        # CHECK: "ext_field_specifiers.operand_specifier"(%c1_i32, %c1_i32) {operandSegmentSizes = array<i32: 1, 0, 1>} : (i32, i32) -> ()
        # CHECK: %0:4 = "ext_field_specifiers.result_specifier"() {resultSegmentSizes = array<i32: 1, 1, 1, 0, 1>} : () -> (i32, i16, i8, i32)
        # CHECK: "ext_field_specifiers.attribute_specifier"() {a = 43 : i32, b = 42 : i32, c = "a", d = 100 : i32} : () -> ()
        print(module)
