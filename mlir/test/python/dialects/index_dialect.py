# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import index, arith


def run(f):
    print("\nTEST:", f.__name__)
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f(ctx)
        print(module)


# CHECK-LABEL: TEST: testConstantOp
@run
def testConstantOp(ctx):
    a = index.ConstantOp(value=42)
    # CHECK: %{{.*}} = index.constant 42


# CHECK-LABEL: TEST: testBoolConstantOp
@run
def testBoolConstantOp(ctx):
    a = index.BoolConstantOp(value=True)
    # CHECK: %{{.*}} = index.bool.constant true


# CHECK-LABEL: TEST: testAndOp
@run
def testAndOp(ctx):
    a = index.ConstantOp(value=42)
    r = index.AndOp(a, a)
    # CHECK: %{{.*}} = index.and %{{.*}}, %{{.*}}


# CHECK-LABEL: TEST: testOrOp
@run
def testOrOp(ctx):
    a = index.ConstantOp(value=42)
    r = index.OrOp(a, a)
    # CHECK: %{{.*}} = index.or %{{.*}}, %{{.*}}


# CHECK-LABEL: TEST: testXOrOp
@run
def testXOrOp(ctx):
    a = index.ConstantOp(value=42)
    r = index.XOrOp(a, a)
    # CHECK: %{{.*}} = index.xor %{{.*}}, %{{.*}}


# CHECK-LABEL: TEST: testCastSOp
@run
def testCastSOp(ctx):
    a = index.ConstantOp(value=42)
    b = arith.ConstantOp(value=23, result=IntegerType.get_signless(64))
    c = index.CastSOp(input=a, output=IntegerType.get_signless(32))
    d = index.CastSOp(input=b, output=IndexType.get())
    # CHECK: %{{.*}} = index.casts %{{.*}} : index to i32
    # CHECK: %{{.*}} = index.casts %{{.*}} : i64 to index


# CHECK-LABEL: TEST: testCastUOp
@run
def testCastUOp(ctx):
    a = index.ConstantOp(value=42)
    b = arith.ConstantOp(value=23, result=IntegerType.get_signless(64))
    c = index.CastUOp(input=a, output=IntegerType.get_signless(32))
    d = index.CastUOp(input=b, output=IndexType.get())
    # CHECK: %{{.*}} = index.castu %{{.*}} : index to i32
    # CHECK: %{{.*}} = index.castu %{{.*}} : i64 to index


# CHECK-LABEL: TEST: testCeilDivSOp
@run
def testCeilDivSOp(ctx):
    a = index.ConstantOp(value=42)
    r = index.CeilDivSOp(a, a)
    # CHECK: %{{.*}} = index.ceildivs %{{.*}}, %{{.*}}


# CHECK-LABEL: TEST: testCeilDivUOp
@run
def testCeilDivUOp(ctx):
    a = index.ConstantOp(value=42)
    r = index.CeilDivUOp(a, a)
    # CHECK: %{{.*}} = index.ceildivu %{{.*}}, %{{.*}}


# CHECK-LABEL: TEST: testCmpOp
@run
def testCmpOp(ctx):
    a = index.ConstantOp(value=42)
    b = index.ConstantOp(value=23)
    pred = AttrBuilder.get("IndexCmpPredicateAttr")("slt", context=ctx)
    r = index.CmpOp(pred, lhs=a, rhs=b)
    # CHECK: %{{.*}} = index.cmp slt(%{{.*}}, %{{.*}})


# CHECK-LABEL: TEST: testAddOp
@run
def testAddOp(ctx):
    a = index.ConstantOp(value=42)
    r = index.AddOp(a, a)
    # CHECK: %{{.*}} = index.add %{{.*}}, %{{.*}}


# CHECK-LABEL: TEST: testSubOp
@run
def testSubOp(ctx):
    a = index.ConstantOp(value=42)
    r = index.SubOp(a, a)
    # CHECK: %{{.*}} = index.sub %{{.*}}, %{{.*}}


# CHECK-LABEL: TEST: testMulOp
@run
def testMulOp(ctx):
    a = index.ConstantOp(value=42)
    r = index.MulOp(a, a)
    # CHECK: %{{.*}} = index.mul %{{.*}}, %{{.*}}


# CHECK-LABEL: TEST: testDivSOp
@run
def testDivSOp(ctx):
    a = index.ConstantOp(value=42)
    r = index.DivSOp(a, a)
    # CHECK: %{{.*}} = index.divs %{{.*}}, %{{.*}}


# CHECK-LABEL: TEST: testDivUOp
@run
def testDivUOp(ctx):
    a = index.ConstantOp(value=42)
    r = index.DivUOp(a, a)
    # CHECK: %{{.*}} = index.divu %{{.*}}, %{{.*}}


# CHECK-LABEL: TEST: testFloorDivSOp
@run
def testFloorDivSOp(ctx):
    a = index.ConstantOp(value=42)
    r = index.FloorDivSOp(a, a)
    # CHECK: %{{.*}} = index.floordivs %{{.*}}, %{{.*}}


# CHECK-LABEL: TEST: testMaxSOp
@run
def testMaxSOp(ctx):
    a = index.ConstantOp(value=42)
    b = index.ConstantOp(value=23)
    r = index.MaxSOp(a, b)
    # CHECK: %{{.*}} = index.maxs %{{.*}}, %{{.*}}


# CHECK-LABEL: TEST: testMaxUOp
@run
def testMaxUOp(ctx):
    a = index.ConstantOp(value=42)
    b = index.ConstantOp(value=23)
    r = index.MaxUOp(a, b)
    # CHECK: %{{.*}} = index.maxu %{{.*}}, %{{.*}}


# CHECK-LABEL: TEST: testMinSOp
@run
def testMinSOp(ctx):
    a = index.ConstantOp(value=42)
    b = index.ConstantOp(value=23)
    r = index.MinSOp(a, b)
    # CHECK: %{{.*}} = index.mins %{{.*}}, %{{.*}}


# CHECK-LABEL: TEST: testMinUOp
@run
def testMinUOp(ctx):
    a = index.ConstantOp(value=42)
    b = index.ConstantOp(value=23)
    r = index.MinUOp(a, b)
    # CHECK: %{{.*}} = index.minu %{{.*}}, %{{.*}}


# CHECK-LABEL: TEST: testRemSOp
@run
def testRemSOp(ctx):
    a = index.ConstantOp(value=42)
    b = index.ConstantOp(value=23)
    r = index.RemSOp(a, b)
    # CHECK: %{{.*}} = index.rems %{{.*}}, %{{.*}}


# CHECK-LABEL: TEST: testRemUOp
@run
def testRemUOp(ctx):
    a = index.ConstantOp(value=42)
    b = index.ConstantOp(value=23)
    r = index.RemUOp(a, b)
    # CHECK: %{{.*}} = index.remu %{{.*}}, %{{.*}}


# CHECK-LABEL: TEST: testShlOp
@run
def testShlOp(ctx):
    a = index.ConstantOp(value=42)
    b = index.ConstantOp(value=3)
    r = index.ShlOp(a, b)
    # CHECK: %{{.*}} = index.shl %{{.*}}, %{{.*}}


# CHECK-LABEL: TEST: testShrSOp
@run
def testShrSOp(ctx):
    a = index.ConstantOp(value=42)
    b = index.ConstantOp(value=3)
    r = index.ShrSOp(a, b)
    # CHECK: %{{.*}} = index.shrs %{{.*}}, %{{.*}}


# CHECK-LABEL: TEST: testShrUOp
@run
def testShrUOp(ctx):
    a = index.ConstantOp(value=42)
    b = index.ConstantOp(value=3)
    r = index.ShrUOp(a, b)
    # CHECK: %{{.*}} = index.shru %{{.*}}, %{{.*}}


# CHECK-LABEL: TEST: testSizeOfOp
@run
def testSizeOfOp(ctx):
    r = index.SizeOfOp()
    # CHECK: %{{.*}} = index.sizeof
