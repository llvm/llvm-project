// RUN: mlir-opt %s -affine-super-vectorize="virtual-vector-size=4" | FileCheck %s

// A non-unit coefficient on the vectorized IV is not a contiguous access. A
// plain vector.transfer_write would touch consecutive cells even though the
// scalar loop writes only every fourth cell.

// CHECK-LABEL: func.func @non_unit_stride_store_unsupported
// CHECK:         affine.for
// CHECK:           affine.store
// CHECK-NOT:     vector.transfer_write
func.func @non_unit_stride_store_unsupported(%arg0: memref<64xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  affine.for %i = 0 to 16 {
    affine.store %cst, %arg0[%i * 4] : memref<64xf32>
  }
  return
}
