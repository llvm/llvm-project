# RUN: %PYTHON %s | FileCheck %s

import gc
from mlir.ir import *

def run(f):
  print("\nTEST:", f.__name__)
  f()
  gc.collect()
  assert Context._get_live_count() == 0


# CHECK-LABEL: TEST: testAffineMapCapsule
def testAffineMapCapsule():
  with Context() as ctx:
    am1 = AffineMap.get_empty(ctx)
  # CHECK: mlir.ir.AffineMap._CAPIPtr
  affine_map_capsule = am1._CAPIPtr
  print(affine_map_capsule)
  am2 = AffineMap._CAPICreate(affine_map_capsule)
  assert am2 == am1
  assert am2.context is ctx

run(testAffineMapCapsule)


# CHECK-LABEL: TEST: testAffineMapGet
def testAffineMapGet():
  with Context() as ctx:
    d0 = AffineDimExpr.get(0)
    d1 = AffineDimExpr.get(1)
    c2 = AffineConstantExpr.get(2)

    # CHECK: (d0, d1)[s0, s1, s2] -> ()
    map0 = AffineMap.get(2, 3, [])
    print(map0)

    # CHECK: (d0, d1)[s0, s1, s2] -> (d1, 2)
    map1 = AffineMap.get(2, 3, [d1, c2])
    print(map1)

    # CHECK: () -> (2)
    map2 = AffineMap.get(0, 0, [c2])
    print(map2)

    # CHECK: (d0, d1) -> (d0, d1)
    map3 = AffineMap.get(2, 0, [d0, d1])
    print(map3)

    # CHECK: (d0, d1) -> (d1)
    map4 = AffineMap.get(2, 0, [d1])
    print(map4)

    # CHECK: (d0, d1, d2) -> (d2, d0, d1)
    map5 = AffineMap.get_permutation([2, 0, 1])
    print(map5)

    assert map1 == AffineMap.get(2, 3, [d1, c2])
    assert AffineMap.get(0, 0, []) == AffineMap.get_empty()
    assert map2 == AffineMap.get_constant(2)
    assert map3 == AffineMap.get_identity(2)
    assert map4 == AffineMap.get_minor_identity(2, 1)

    try:
      AffineMap.get(1, 1, [1])
    except RuntimeError as e:
      # CHECK: Invalid expression when attempting to create an AffineMap
      print(e)

    try:
      AffineMap.get(1, 1, [None])
    except RuntimeError as e:
      # CHECK: Invalid expression (None?) when attempting to create an AffineMap
      print(e)

    try:
      AffineMap.get_permutation([1, 0, 1])
    except RuntimeError as e:
      # CHECK: Invalid permutation when attempting to create an AffineMap
      print(e)

    try:
      map3.get_submap([42])
    except ValueError as e:
      # CHECK: result position out of bounds
      print(e)

    try:
      map3.get_minor_submap(42)
    except ValueError as e:
      # CHECK: number of results out of bounds
      print(e)

    try:
      map3.get_major_submap(42)
    except ValueError as e:
      # CHECK: number of results out of bounds
      print(e)

run(testAffineMapGet)


# CHECK-LABEL: TEST: testAffineMapDerive
def testAffineMapDerive():
  with Context() as ctx:
    map5 = AffineMap.get_identity(5)

    # CHECK: (d0, d1, d2, d3, d4) -> (d1, d2, d3)
    map123 = map5.get_submap([1,2,3])
    print(map123)

    # CHECK: (d0, d1, d2, d3, d4) -> (d0, d1)
    map01 = map5.get_major_submap(2)
    print(map01)

    # CHECK: (d0, d1, d2, d3, d4) -> (d3, d4)
    map34 = map5.get_minor_submap(2)
    print(map34)

run(testAffineMapDerive)


# CHECK-LABEL: TEST: testAffineMapProperties
def testAffineMapProperties():
  with Context():
    d0 = AffineDimExpr.get(0)
    d1 = AffineDimExpr.get(1)
    d2 = AffineDimExpr.get(2)
    map1 = AffineMap.get(3, 0, [d2, d0])
    map2 = AffineMap.get(3, 0, [d2, d0, d1])
    map3 = AffineMap.get(3, 1, [d2, d0, d1])
    # CHECK: False
    print(map1.is_permutation)
    # CHECK: True
    print(map1.is_projected_permutation)
    # CHECK: True
    print(map2.is_permutation)
    # CHECK: True
    print(map2.is_projected_permutation)
    # CHECK: False
    print(map3.is_permutation)
    # CHECK: False
    print(map3.is_projected_permutation)

run(testAffineMapProperties)


# CHECK-LABEL: TEST: testAffineMapExprs
def testAffineMapExprs():
  with Context():
    d0 = AffineDimExpr.get(0)
    d1 = AffineDimExpr.get(1)
    d2 = AffineDimExpr.get(2)
    map3 = AffineMap.get(3, 1, [d2, d0, d1])

    # CHECK: 3
    print(map3.n_dims)
    # CHECK: 4
    print(map3.n_inputs)
    # CHECK: 1
    print(map3.n_symbols)
    assert map3.n_inputs == map3.n_dims + map3.n_symbols

    # CHECK: 3
    print(len(map3.results))
    for expr in map3.results:
      # CHECK: d2
      # CHECK: d0
      # CHECK: d1
      print(expr)
    for expr in map3.results[-1:-4:-1]:
      # CHECK: d1
      # CHECK: d0
      # CHECK: d2
      print(expr)
    assert list(map3.results) == [d2, d0, d1]

run(testAffineMapExprs)
