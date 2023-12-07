# RUN: %PYTHON %s | FileCheck %s

import gc
from mlir.ir import *


def run(f):
    print("\nTEST:", f.__name__)
    f()
    gc.collect()
    assert Context._get_live_count() == 0
    return f


# CHECK-LABEL: TEST: testAffineExprCapsule
@run
def testAffineExprCapsule():
    with Context() as ctx:
        affine_expr = AffineExpr.get_constant(42)

    affine_expr_capsule = affine_expr._CAPIPtr
    # CHECK: capsule object
    # CHECK: mlir.ir.AffineExpr._CAPIPtr
    print(affine_expr_capsule)

    affine_expr_2 = AffineExpr._CAPICreate(affine_expr_capsule)
    assert affine_expr == affine_expr_2
    assert affine_expr_2.context == ctx


# CHECK-LABEL: TEST: testAffineExprEq
@run
def testAffineExprEq():
    with Context():
        a1 = AffineExpr.get_constant(42)
        a2 = AffineExpr.get_constant(42)
        a3 = AffineExpr.get_constant(43)
        # CHECK: True
        print(a1 == a1)
        # CHECK: True
        print(a1 == a2)
        # CHECK: False
        print(a1 == a3)
        # CHECK: False
        print(a1 == None)
        # CHECK: False
        print(a1 == "foo")


# CHECK-LABEL: TEST: testAffineExprContext
@run
def testAffineExprContext():
    with Context():
        a1 = AffineExpr.get_constant(42)
    with Context():
        a2 = AffineExpr.get_constant(42)

    # CHECK: False
    print(a1 == a2)


run(testAffineExprContext)


# CHECK-LABEL: TEST: testAffineExprConstant
@run
def testAffineExprConstant():
    with Context():
        a1 = AffineExpr.get_constant(42)
        # CHECK: 42
        print(a1.value)
        # CHECK: 42
        print(a1)

        a2 = AffineConstantExpr.get(42)
        # CHECK: 42
        print(a2.value)
        # CHECK: 42
        print(a2)

        assert a1 == a2


# CHECK-LABEL: TEST: testAffineExprDim
@run
def testAffineExprDim():
    with Context():
        d1 = AffineExpr.get_dim(1)
        d11 = AffineDimExpr.get(1)
        d2 = AffineDimExpr.get(2)

        # CHECK: 1
        print(d1.position)
        # CHECK: d1
        print(d1)

        # CHECK: 2
        print(d2.position)
        # CHECK: d2
        print(d2)

        assert d1 == d11
        assert d1 != d2


# CHECK-LABEL: TEST: testAffineExprSymbol
@run
def testAffineExprSymbol():
    with Context():
        s1 = AffineExpr.get_symbol(1)
        s11 = AffineSymbolExpr.get(1)
        s2 = AffineSymbolExpr.get(2)

        # CHECK: 1
        print(s1.position)
        # CHECK: s1
        print(s1)

        # CHECK: 2
        print(s2.position)
        # CHECK: s2
        print(s2)

        assert s1 == s11
        assert s1 != s2


# CHECK-LABEL: TEST: testAffineAddExpr
@run
def testAffineAddExpr():
    with Context():
        d1 = AffineDimExpr.get(1)
        d2 = AffineDimExpr.get(2)
        d12 = AffineExpr.get_add(d1, d2)
        # CHECK: d1 + d2
        print(d12)

        d12op = d1 + d2
        # CHECK: d1 + d2
        print(d12op)

        d1cst_op = d1 + 2
        # CHECK: d1 + 2
        print(d1cst_op)

        d1cst_op2 = 2 + d1
        # CHECK: d1 + 2
        print(d1cst_op2)

        assert d12 == d12op
        assert d12.lhs == d1
        assert d12.rhs == d2


# CHECK-LABEL: TEST: testAffineMulExpr
@run
def testAffineMulExpr():
    with Context():
        d1 = AffineDimExpr.get(1)
        c2 = AffineConstantExpr.get(2)
        expr = AffineExpr.get_mul(d1, c2)
        # CHECK: d1 * 2
        print(expr)

        # CHECK: d1 * 2
        op = d1 * c2
        print(op)

        # CHECK: d1 * 2
        op_cst = d1 * 2
        print(op_cst)

        # CHECK: d1 * 2
        op_cst2 = 2 * d1
        print(op_cst2)

        assert expr == op
        assert expr == op_cst
        assert expr.lhs == d1
        assert expr.rhs == c2


# CHECK-LABEL: TEST: testAffineModExpr
@run
def testAffineModExpr():
    with Context():
        d1 = AffineDimExpr.get(1)
        c2 = AffineConstantExpr.get(2)
        expr = AffineExpr.get_mod(d1, c2)
        # CHECK: d1 mod 2
        print(expr)

        # CHECK: d1 mod 2
        op = d1 % c2
        print(op)

        # CHECK: d1 mod 2
        op_cst = d1 % 2
        print(op_cst)

        # CHECK: 2 mod d1
        print(2 % d1)

        assert expr == op
        assert expr == op_cst
        assert expr.lhs == d1
        assert expr.rhs == c2

        expr2 = AffineExpr.get_mod(c2, d1)
        expr3 = AffineExpr.get_mod(2, d1)
        expr4 = AffineExpr.get_mod(d1, 2)

        # CHECK: 2 mod d1
        print(expr2)
        # CHECK: 2 mod d1
        print(expr3)
        # CHECK: d1 mod 2
        print(expr4)

        assert expr2 == expr3
        assert expr4 == expr


# CHECK-LABEL: TEST: testAffineFloorDivExpr
@run
def testAffineFloorDivExpr():
    with Context():
        d1 = AffineDimExpr.get(1)
        c2 = AffineConstantExpr.get(2)
        expr = AffineExpr.get_floor_div(d1, c2)
        # CHECK: d1 floordiv 2
        print(expr)

        assert expr.lhs == d1
        assert expr.rhs == c2

        expr2 = AffineExpr.get_floor_div(c2, d1)
        expr3 = AffineExpr.get_floor_div(2, d1)
        expr4 = AffineExpr.get_floor_div(d1, 2)

        # CHECK: 2 floordiv d1
        print(expr2)
        # CHECK: 2 floordiv d1
        print(expr3)
        # CHECK: d1 floordiv 2
        print(expr4)

        assert expr2 == expr3
        assert expr4 == expr


# CHECK-LABEL: TEST: testAffineCeilDivExpr
@run
def testAffineCeilDivExpr():
    with Context():
        d1 = AffineDimExpr.get(1)
        c2 = AffineConstantExpr.get(2)
        expr = AffineExpr.get_ceil_div(d1, c2)
        # CHECK: d1 ceildiv 2
        print(expr)

        assert expr.lhs == d1
        assert expr.rhs == c2

        expr2 = AffineExpr.get_ceil_div(c2, d1)
        expr3 = AffineExpr.get_ceil_div(2, d1)
        expr4 = AffineExpr.get_ceil_div(d1, 2)

        # CHECK: 2 ceildiv d1
        print(expr2)
        # CHECK: 2 ceildiv d1
        print(expr3)
        # CHECK: d1 ceildiv 2
        print(expr4)

        assert expr2 == expr3
        assert expr4 == expr


# CHECK-LABEL: TEST: testAffineExprSub
@run
def testAffineExprSub():
    with Context():
        d1 = AffineDimExpr.get(1)
        d2 = AffineDimExpr.get(2)
        expr = d1 - d2
        # CHECK: d1 - d2
        print(expr)

        assert expr.lhs == d1
        rhs = AffineMulExpr(expr.rhs)
        # CHECK: d2
        print(rhs.lhs)
        # CHECK: -1
        print(rhs.rhs)

        # CHECK: d1 - 42
        print(d1 - 42)
        # CHECK: -d1 + 42
        print(42 - d1)

        c42 = AffineConstantExpr.get(42)
        assert d1 - 42 == d1 - c42
        assert 42 - d1 == c42 - d1


# CHECK-LABEL: TEST: testClassHierarchy
@run
def testClassHierarchy():
    with Context():
        d1 = AffineDimExpr.get(1)
        c2 = AffineConstantExpr.get(2)
        add = AffineAddExpr.get(d1, c2)
        mul = AffineMulExpr.get(d1, c2)
        mod = AffineModExpr.get(d1, c2)
        floor_div = AffineFloorDivExpr.get(d1, c2)
        ceil_div = AffineCeilDivExpr.get(d1, c2)

        # CHECK: False
        print(isinstance(d1, AffineBinaryExpr))
        # CHECK: False
        print(isinstance(c2, AffineBinaryExpr))
        # CHECK: True
        print(isinstance(add, AffineBinaryExpr))
        # CHECK: True
        print(isinstance(mul, AffineBinaryExpr))
        # CHECK: True
        print(isinstance(mod, AffineBinaryExpr))
        # CHECK: True
        print(isinstance(floor_div, AffineBinaryExpr))
        # CHECK: True
        print(isinstance(ceil_div, AffineBinaryExpr))

        try:
            AffineBinaryExpr(d1)
        except ValueError as e:
            # CHECK: Cannot cast affine expression to AffineBinaryExpr
            print(e)

        try:
            AffineBinaryExpr(c2)
        except ValueError as e:
            # CHECK: Cannot cast affine expression to AffineBinaryExpr
            print(e)


# CHECK-LABEL: TEST: testIsInstance
@run
def testIsInstance():
    with Context():
        d1 = AffineDimExpr.get(1)
        c2 = AffineConstantExpr.get(2)
        add = AffineAddExpr.get(d1, c2)
        mul = AffineMulExpr.get(d1, c2)

        # CHECK: True
        print(AffineDimExpr.isinstance(d1))
        # CHECK: False
        print(AffineConstantExpr.isinstance(d1))
        # CHECK: True
        print(AffineConstantExpr.isinstance(c2))
        # CHECK: False
        print(AffineMulExpr.isinstance(c2))
        # CHECK: True
        print(AffineAddExpr.isinstance(add))
        # CHECK: False
        print(AffineMulExpr.isinstance(add))
        # CHECK: True
        print(AffineMulExpr.isinstance(mul))
        # CHECK: False
        print(AffineAddExpr.isinstance(mul))


# CHECK-LABEL: TEST: testCompose
@run
def testCompose():
    with Context():
        # d0 + d2.
        expr = AffineAddExpr.get(AffineDimExpr.get(0), AffineDimExpr.get(2))

        # (d0, d1, d2)[s0, s1] -> (d0 + s1, d1 + s0, d0 + d1 + d2)
        map1 = AffineAddExpr.get(AffineDimExpr.get(0), AffineSymbolExpr.get(1))
        map2 = AffineAddExpr.get(AffineDimExpr.get(1), AffineSymbolExpr.get(0))
        map3 = AffineAddExpr.get(
            AffineAddExpr.get(AffineDimExpr.get(0), AffineDimExpr.get(1)),
            AffineDimExpr.get(2),
        )
        map = AffineMap.get(3, 2, [map1, map2, map3])

        # CHECK: d0 + s1 + d0 + d1 + d2
        print(expr.compose(map))


# CHECK-LABEL: TEST: testHash
@run
def testHash():
    with Context():
        d0 = AffineDimExpr.get(0)
        s1 = AffineSymbolExpr.get(1)
        assert hash(d0) == hash(AffineDimExpr.get(0))
        assert hash(d0 + s1) == hash(AffineAddExpr.get(d0, s1))

        dictionary = dict()
        dictionary[d0] = 0
        dictionary[s1] = 1
        assert d0 in dictionary
        assert s1 in dictionary
