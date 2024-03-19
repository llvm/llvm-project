# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import index, arith


def run(f):
    print("\nTEST:", f.__name__)
    f()


# CHECK-LABEL: TEST: testConstantOp
@run
def testConstantOp():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            a = index.ConstantOp(value=42)
        # CHECK: %[[A:.*]] = index.constant 42
        print(module)


# CHECK-LABEL: TEST: testBoolConstantOp
@run
def testBoolConstantOp():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            a = index.BoolConstantOp(value=True)
        # CHECK: %[[A:.*]] = index.bool.constant true
        print(module)


# CHECK-LABEL: TEST: testAndOp
@run
def testAndOp():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            a = index.ConstantOp(value=42)
            r = index.AndOp(a, a)
        # CHECK: %[[R:.*]] = index.and {{.*}}, {{.*}}
        print(module)


# CHECK-LABEL: TEST: testOrOp
@run
def testOrOp():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            a = index.ConstantOp(value=42)
            r = index.OrOp(a, a)
        # CHECK: %[[R:.*]] = index.or {{.*}}, {{.*}}
        print(module)


# CHECK-LABEL: TEST: testXOrOp
@run
def testXOrOp():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            a = index.ConstantOp(value=42)
            r = index.XOrOp(a, a)
        # CHECK: %[[R:.*]] = index.xor {{.*}}, {{.*}}
        print(module)


# CHECK-LABEL: TEST: testCastSOp
@run
def testCastSOp():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            a = index.ConstantOp(value=42)
            b = arith.ConstantOp(value=23, result=IntegerType.get_signless(64))
            c = index.CastSOp(input=a, output=IntegerType.get_signless(32))
            d = index.CastSOp(input=b, output=IndexType.get())
        # CHECK: %[[C:.*]] = index.casts {{.*}} : index to i32
        # CHECK: %[[D:.*]] = index.casts {{.*}} : i64 to index
        print(module)


# CHECK-LABEL: TEST: testCastUOp
@run
def testCastUOp():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            a = index.ConstantOp(value=42)
            b = arith.ConstantOp(value=23, result=IntegerType.get_signless(64))
            c = index.CastUOp(input=a, output=IntegerType.get_signless(32))
            d = index.CastUOp(input=b, output=IndexType.get())
        # CHECK: %[[C:.*]] = index.castu {{.*}} : index to i32
        # CHECK: %[[D:.*]] = index.castu {{.*}} : i64 to index
        print(module)


# CHECK-LABEL: TEST: testCeilDivSOp
@run
def testCeilDivSOp():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            a = index.ConstantOp(value=42)
            r = index.CeilDivSOp(a, a)
        # CHECK: %[[R:.*]] = index.ceildivs {{.*}}, {{.*}}
        print(module)


# CHECK-LABEL: TEST: testCeilDivUOp
@run
def testCeilDivUOp():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            a = index.ConstantOp(value=42)
            r = index.CeilDivUOp(a, a)
        # CHECK: %[[R:.*]] = index.ceildivu {{.*}}, {{.*}}
        print(module)


# CHECK-LABEL: TEST: testCmpOp
@run
def testCmpOp():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            a = index.ConstantOp(value=42)
            b = index.ConstantOp(value=23)
            pred = AttrBuilder.get("IndexCmpPredicateAttr")("slt", context=ctx)
            r = index.CmpOp(pred, lhs=a, rhs=b)
        # CHECK: %[[R:.*]] = index.cmp slt({{.*}}, {{.*}})
        print(module)


# CHECK-LABEL: TEST: testAddOp
@run
def testAddOp():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            a = index.ConstantOp(value=42)
            r = index.AddOp(a, a)
        # CHECK: %[[R:.*]] = index.add {{.*}}, {{.*}}
        print(module)


# CHECK-LABEL: TEST: testSubOp
@run
def testSubOp():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            a = index.ConstantOp(value=42)
            r = index.SubOp(a, a)
        # CHECK: %[[R:.*]] = index.sub {{.*}}, {{.*}}
        print(module)


# CHECK-LABEL: TEST: testMulOp
@run
def testMulOp():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            a = index.ConstantOp(value=42)
            r = index.MulOp(a, a)
        # CHECK: %[[R:.*]] = index.mul {{.*}}, {{.*}}
        print(module)


# CHECK-LABEL: TEST: testDivSOp
@run
def testDivSOp():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            a = index.ConstantOp(value=42)
            r = index.DivSOp(a, a)
        # CHECK: %[[R:.*]] = index.divs {{.*}}, {{.*}}
        print(module)


# CHECK-LABEL: TEST: testDivUOp
@run
def testDivUOp():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            a = index.ConstantOp(value=42)
            r = index.DivUOp(a, a)
        # CHECK: %[[R:.*]] = index.divu {{.*}}, {{.*}}
        print(module)


# CHECK-LABEL: TEST: testFloorDivSOp
@run
def testFloorDivSOp():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            a = index.ConstantOp(value=42)
            r = index.FloorDivSOp(a, a)
        # CHECK: %[[R:.*]] = index.floordivs {{.*}}, {{.*}}
        print(module)


# CHECK-LABEL: TEST: testMaxSOp
@run
def testMaxSOp():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            a = index.ConstantOp(value=42)
            b = index.ConstantOp(value=23)
            r = index.MaxSOp(a, b)
        # CHECK: %[[R:.*]] = index.maxs {{.*}}, {{.*}}
        print(module)


# CHECK-LABEL: TEST: testMaxUOp
@run
def testMaxUOp():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            a = index.ConstantOp(value=42)
            b = index.ConstantOp(value=23)
            r = index.MaxUOp(a, b)
        # CHECK: %[[R:.*]] = index.maxu {{.*}}, {{.*}}
        print(module)


# CHECK-LABEL: TEST: testMinSOp
@run
def testMinSOp():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            a = index.ConstantOp(value=42)
            b = index.ConstantOp(value=23)
            r = index.MinSOp(a, b)
        # CHECK: %[[R:.*]] = index.mins {{.*}}, {{.*}}
        print(module)


# CHECK-LABEL: TEST: testMinUOp
@run
def testMinUOp():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            a = index.ConstantOp(value=42)
            b = index.ConstantOp(value=23)
            r = index.MinUOp(a, b)
        # CHECK: %[[R:.*]] = index.minu {{.*}}, {{.*}}
        print(module)


# CHECK-LABEL: TEST: testRemSOp
@run
def testRemSOp():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            a = index.ConstantOp(value=42)
            b = index.ConstantOp(value=23)
            r = index.RemSOp(a, b)
        # CHECK: %[[R:.*]] = index.rems {{.*}}, {{.*}}
        print(module)


# CHECK-LABEL: TEST: testRemUOp
@run
def testRemUOp():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            a = index.ConstantOp(value=42)
            b = index.ConstantOp(value=23)
            r = index.RemUOp(a, b)
        # CHECK: %[[R:.*]] = index.remu {{.*}}, {{.*}}
        print(module)


# CHECK-LABEL: TEST: testShlOp
@run
def testShlOp():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            a = index.ConstantOp(value=42)
            b = index.ConstantOp(value=3)
            r = index.ShlOp(a, b)
        # CHECK: %[[R:.*]] = index.shl {{.*}}, {{.*}}
        print(module)


# CHECK-LABEL: TEST: testShrSOp
@run
def testShrSOp():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            a = index.ConstantOp(value=42)
            b = index.ConstantOp(value=3)
            r = index.ShrSOp(a, b)
        # CHECK: %[[R:.*]] = index.shrs {{.*}}, {{.*}}
        print(module)


# CHECK-LABEL: TEST: testShrUOp
@run
def testShrUOp():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            a = index.ConstantOp(value=42)
            b = index.ConstantOp(value=3)
            r = index.ShrUOp(a, b)
        # CHECK: %[[R:.*]] = index.shru {{.*}}, {{.*}}
        print(module)


# CHECK-LABEL: TEST: testSizeOfOp
@run
def testSizeOfOp():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            r = index.SizeOfOp()
        # CHECK: %[[R:.*]] = index.sizeof
        print(module)
