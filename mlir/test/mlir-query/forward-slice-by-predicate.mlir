// RUN: mlir-query %s -c "m getUsersByPredicate(anyOf(hasOpName(\"memref.alloc\"),isConstantOp()),anyOf(hasOpName(\"affine.load\"), hasOpName(\"memref.dealloc\")),true)" | FileCheck %s

func.func @slice_depth1_loop_nest_with_offsets() {
  %0 = memref.alloc() : memref<100xf32>
  %cst = arith.constant 7.000000e+00 : f32
  affine.for %i0 = 0 to 16 {
    %a0 = affine.apply affine_map<(d0) -> (d0 + 2)>(%i0)
    affine.store %cst, %0[%a0] : memref<100xf32>
  }
  affine.for %i1 = 4 to 8 {
    %a1 = affine.apply affine_map<(d0) -> (d0 - 1)>(%i1)
    %1 = affine.load %0[%a1] : memref<100xf32>
  }
  return
}

// CHECK: Match #1:
// CHECK: {{.*}}.mlir:4:8: note: "root" binds here
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<100xf32>

// CHECK: affine.store %cst, %0[%a0] : memref<100xf32>

// CHECK: Match #2:
// CHECK: {{.*}}.mlir:5:10: note: "root" binds here
// CHECK: %[[CST:.*]] = arith.constant 7.000000e+00 : f32

// CHECK: affine.store %[[CST]], %0[%a0] : memref<100xf32>
