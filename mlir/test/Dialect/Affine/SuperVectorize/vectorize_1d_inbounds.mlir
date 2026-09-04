// RUN: mlir-opt %s -affine-super-vectorize="virtual-vector-size=8" -split-input-file | FileCheck %s

// The dimension is divisible by the vector size, but the index is offset by
// one, so the last vector runs past the end of the memref.

// CHECK-LABEL: func.func @offset_index
//       CHECK:   affine.for %{{.*}} = 0 to 15 step 8 {
//       CHECK:     vector.transfer_read %{{.*}}[%{{.*}}], %{{[0-9]+}} : memref<16xf32>, vector<8xf32>
//       CHECK:     vector.transfer_write %{{.*}}, %{{.*}}[%{{[a-z0-9]+}}] : vector<8xf32>, memref<16xf32>
func.func @offset_index(%A: memref<16xf32>, %B: memref<16xf32>) {
  affine.for %i = 0 to 15 {
    %v = affine.load %A[%i + 1] : memref<16xf32>
    affine.store %v, %B[%i + 1] : memref<16xf32>
  }
  return
}

// -----

// An offset that is a multiple of the vector size keeps the accesses aligned.

// CHECK-LABEL: func.func @aligned_offset_index
//       CHECK:   affine.for %{{.*}} = 0 to 8 step 8 {
//       CHECK:     vector.transfer_read %{{.*}}[%{{.*}}], %{{.*}} {in_bounds = [true]} : memref<16xf32>, vector<8xf32>
//       CHECK:     vector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}] {in_bounds = [true]} : vector<8xf32>, memref<16xf32>
func.func @aligned_offset_index(%A: memref<16xf32>, %B: memref<16xf32>) {
  affine.for %i = 0 to 8 {
    %v = affine.load %A[%i + 8] : memref<16xf32>
    affine.store %v, %B[%i + 8] : memref<16xf32>
  }
  return
}

// -----

// A lower bound that is not a multiple of the vector size shifts every vector
// off alignment.

// CHECK-LABEL: func.func @unaligned_lower_bound
//       CHECK:   affine.for %{{.*}} = 4 to 16 step 8 {
//       CHECK:     vector.transfer_read %{{.*}}[%{{.*}}], %{{[0-9]+}} : memref<16xf32>, vector<8xf32>
//       CHECK:     vector.transfer_write %{{.*}}, %{{.*}}[%{{[a-z0-9]+}}] : vector<8xf32>, memref<16xf32>
func.func @unaligned_lower_bound(%A: memref<16xf32>, %B: memref<16xf32>) {
  affine.for %i = 4 to 16 {
    %v = affine.load %A[%i] : memref<16xf32>
    affine.store %v, %B[%i] : memref<16xf32>
  }
  return
}

// -----

// CHECK-LABEL: func.func @aligned_lower_bound
//       CHECK:   affine.for %{{.*}} = 8 to 16 step 8 {
//       CHECK:     vector.transfer_read %{{.*}}[%{{.*}}], %{{.*}} {in_bounds = [true]} : memref<16xf32>, vector<8xf32>
//       CHECK:     vector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}] {in_bounds = [true]} : vector<8xf32>, memref<16xf32>
func.func @aligned_lower_bound(%A: memref<16xf32>, %B: memref<16xf32>) {
  affine.for %i = 8 to 16 {
    %v = affine.load %A[%i] : memref<16xf32>
    affine.store %v, %B[%i] : memref<16xf32>
  }
  return
}

// -----

// The lower bound of the vectorized loop is the induction variable of a loop
// stepping by a multiple of the vector size, as after tiling.

// CHECK-LABEL: func.func @lower_bound_from_outer_loop
//       CHECK:   affine.for %{{.*}} = 0 to 32 step 16 {
//       CHECK:     affine.for %{{.*}} = #{{.*}}(%{{.*}}) to #{{.*}}(%{{.*}}) step 8 {
//       CHECK:       vector.transfer_read %{{.*}}[%{{.*}}], %{{.*}} {in_bounds = [true]} : memref<32xf32>, vector<8xf32>
//       CHECK:       vector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}] {in_bounds = [true]} : vector<8xf32>, memref<32xf32>
func.func @lower_bound_from_outer_loop(%A: memref<32xf32>, %B: memref<32xf32>) {
  affine.for %i = 0 to 32 step 16 {
    affine.for %ii = affine_map<(d0) -> (d0)>(%i) to affine_map<(d0) -> (d0 + 16)>(%i) {
      %v = affine.load %A[%ii] : memref<32xf32>
      affine.store %v, %B[%ii] : memref<32xf32>
    }
  }
  return
}

// -----

// Same as above, but the outer loop steps by 12, so the inner lower bound is
// not a multiple of the vector size.

// CHECK-LABEL: func.func @unaligned_lower_bound_from_outer_loop
//       CHECK:   affine.for %{{.*}} = 0 to 24 step 12 {
//       CHECK:     affine.for %{{.*}} = #{{.*}}(%{{.*}}) to #{{.*}}(%{{.*}}) step 8 {
//       CHECK:       vector.transfer_read %{{.*}}[%{{.*}}], %{{[0-9]+}} : memref<24xf32>, vector<8xf32>
//       CHECK:       vector.transfer_write %{{.*}}, %{{.*}}[%{{[a-z0-9]+}}] : vector<8xf32>, memref<24xf32>
func.func @unaligned_lower_bound_from_outer_loop(%A: memref<24xf32>, %B: memref<24xf32>) {
  affine.for %i = 0 to 24 step 12 {
    affine.for %ii = affine_map<(d0) -> (d0)>(%i) to affine_map<(d0) -> (d0 + 12)>(%i) {
      %v = affine.load %A[%ii] : memref<24xf32>
      affine.store %v, %B[%ii] : memref<24xf32>
    }
  }
  return
}
