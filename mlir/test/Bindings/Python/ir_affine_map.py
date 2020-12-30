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
