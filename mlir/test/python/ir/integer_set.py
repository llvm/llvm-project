# RUN: %PYTHON %s | FileCheck %s

import gc
from mlir.ir import *

def run(f):
  print("\nTEST:", f.__name__)
  f()
  gc.collect()
  assert Context._get_live_count() == 0


# CHECK-LABEL: TEST: testIntegerSetCapsule
def testIntegerSetCapsule():
  with Context() as ctx:
    is1 = IntegerSet.get_empty(1, 1, ctx)
  capsule = is1._CAPIPtr
  # CHECK: mlir.ir.IntegerSet._CAPIPtr
  print(capsule)
  is2 = IntegerSet._CAPICreate(capsule)
  assert is1 == is2
  assert is2.context is ctx

run(testIntegerSetCapsule)


# CHECK-LABEL: TEST: testIntegerSetGet
def testIntegerSetGet():
  with Context():
    d0 = AffineDimExpr.get(0)
    d1 = AffineDimExpr.get(1)
    s0 = AffineSymbolExpr.get(0)
    c42 = AffineConstantExpr.get(42)

    # CHECK: (d0, d1)[s0] : (d0 - d1 == 0, s0 - 42 >= 0)
    set0 = IntegerSet.get(2, 1, [d0 - d1, s0 - c42], [True, False])
    print(set0)

    # CHECK: (d0)[s0] : (1 == 0)
    set1 = IntegerSet.get_empty(1, 1)
    print(set1)

    # CHECK: (d0)[s0, s1] : (d0 - s1 == 0, s0 - 42 >= 0)
    set2 = set0.get_replaced([d0, AffineSymbolExpr.get(1)], [s0], 1, 2)
    print(set2)

    try:
      IntegerSet.get(2, 1, [], [])
    except ValueError as e:
      # CHECK: Expected non-empty list of constraints
      print(e)

    try:
      IntegerSet.get(2, 1, [d0 - d1], [True, False])
    except ValueError as e:
      # CHECK: Expected the number of constraints to match that of equality flags
      print(e)

    try:
      IntegerSet.get(2, 1, [0], [True])
    except RuntimeError as e:
      # CHECK: Invalid expression when attempting to create an IntegerSet
      print(e)

    try:
      IntegerSet.get(2, 1, [None], [True])
    except RuntimeError as e:
      # CHECK: Invalid expression (None?) when attempting to create an IntegerSet
      print(e)

    try:
      set0.get_replaced([d0], [s0], 1, 1)
    except ValueError as e:
      # CHECK: Expected the number of dimension replacement expressions to match that of dimensions
      print(e)

    try:
      set0.get_replaced([d0, d1], [s0, s0], 1, 1)
    except ValueError as e:
      # CHECK: Expected the number of symbol replacement expressions to match that of symbols
      print(e)

    try:
      set0.get_replaced([d0, 1], [s0], 1, 1)
    except RuntimeError as e:
      # CHECK: Invalid expression when attempting to create an IntegerSet by replacing dimensions
      print(e)

    try:
      set0.get_replaced([d0, d1], [None], 1, 1)
    except RuntimeError as e:
      # CHECK: Invalid expression (None?) when attempting to create an IntegerSet by replacing symbols
      print(e)

run(testIntegerSetGet)


# CHECK-LABEL: TEST: testIntegerSetProperties
def testIntegerSetProperties():
  with Context():
    d0 = AffineDimExpr.get(0)
    d1 = AffineDimExpr.get(1)
    s0 = AffineSymbolExpr.get(0)
    c42 = AffineConstantExpr.get(42)

    set0 = IntegerSet.get(2, 1, [d0 - d1, s0 - c42, s0 - d0], [True, False, False])
    # CHECK: 2
    print(set0.n_dims)
    # CHECK: 1
    print(set0.n_symbols)
    # CHECK: 3
    print(set0.n_inputs)
    # CHECK: 1
    print(set0.n_equalities)
    # CHECK: 2
    print(set0.n_inequalities)

    # CHECK: 3
    print(len(set0.constraints))

    # CHECK-DAG: d0 - d1 == 0
    # CHECK-DAG: s0 - 42 >= 0
    # CHECK-DAG: -d0 + s0 >= 0
    for cstr in set0.constraints:
      print(cstr.expr, end='')
      print(" == 0" if cstr.is_eq else " >= 0")

run(testIntegerSetProperties)
