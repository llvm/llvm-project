// RUN: mlir-opt -verify-diagnostics -ownership-based-buffer-deallocation -split-input-file %s

// Test Case: ownership-based-buffer-deallocation should not fail
//            with cf.assert op

// CHECK-LABEL: func @func_with_assert(
//       CHECK: %0 = arith.cmpi slt, %arg0, %arg1 : index
//       CHECK: cf.assert %0, "%arg0 must be less than %arg1"
func.func @func_with_assert(%arg0: index, %arg1: index) {
  %0 = arith.cmpi slt, %arg0, %arg1 : index
  cf.assert %0, "%arg0 must be less than %arg1"
  return
}

// CHECK-LABEL: func @func_with_assume_alignment(
//       CHECK: %0 = memref.assume_alignment %arg0, 64 : memref<128xi8>
func.func @func_with_assume_alignment(%arg0: memref<128xi8>) {
  %0 = memref.assume_alignment %arg0, 64 : memref<128xi8>
  return
}

// CHECK-LABEL: func @func_with_prefetch(
//       CHECK: memref.prefetch %arg0[%c0, %c0], read, locality<1>, data : memref<4x8xf32>
func.func @func_with_prefetch(%arg0: memref<4x8xf32>) {
  %c0 = arith.constant 0 : index
  memref.prefetch %arg0[%c0, %c0], read, locality<1>, data : memref<4x8xf32>
  return
}
