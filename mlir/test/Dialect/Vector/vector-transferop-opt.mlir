// RUN: mlir-opt %s -test-vector-transferop-opt | FileCheck %s

// CHECK-LABEL: func @forward_dead_store
//   CHECK-NOT:   vector.transfer_write
//   CHECK-NOT:   vector.transfer_read
//       CHECK:   scf.for
//       CHECK:   }
//       CHECK:   vector.transfer_write
//       CHECK:   return
func.func @forward_dead_store(%arg0: i1, %arg1 : memref<4x4xf32>,
  %v0 : vector<1x4xf32>, %v1 : vector<1x4xf32>, %i : index) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  vector.transfer_write %v0, %arg1[%c1, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, memref<4x4xf32>
  %0 = vector.transfer_read %arg1[%c1, %c0], %cf0 {in_bounds = [true, true]} :
    memref<4x4xf32>, vector<1x4xf32>
  %x = scf.for %i0 = %c0 to %c4 step %c1 iter_args(%acc = %0)
    -> (vector<1x4xf32>) {
    %1 = arith.addf %acc, %acc : vector<1x4xf32>
    scf.yield %1 : vector<1x4xf32>
  }
  vector.transfer_write %x, %arg1[%c1, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, memref<4x4xf32>
  return
}

// CHECK-LABEL: func @forward_nested
//       CHECK:   vector.transfer_write
//       CHECK:   vector.transfer_write
//       CHECK:   scf.if
//   CHECK-NOT:     vector.transfer_read
//       CHECK:   }
//       CHECK:   vector.transfer_write
//       CHECK:   return
func.func @forward_nested(%arg0: i1, %arg1 : memref<4x4xf32>, %v0 : vector<1x4xf32>,
  %v1 : vector<1x4xf32>, %i : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cf0 = arith.constant 0.0 : f32
  vector.transfer_write %v1, %arg1[%i, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, memref<4x4xf32>
  vector.transfer_write %v0, %arg1[%c1, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, memref<4x4xf32>
  %x = scf.if %arg0 -> (vector<1x4xf32>) {
    %0 = vector.transfer_read %arg1[%c1, %c0], %cf0 {in_bounds = [true, true]} :
      memref<4x4xf32>, vector<1x4xf32>
    scf.yield %0 : vector<1x4xf32>
  } else {
    scf.yield %v1 : vector<1x4xf32>
  }
  vector.transfer_write %x, %arg1[%c0, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, memref<4x4xf32>
  return
}

// Negative test, the transfer_write in the scf.if region block the store to
// load forwarding because we don't recursively look into the region to realize
// that the transfer_write cannot reach the transfer_read.
// CHECK-LABEL: func @forward_nested_negative
//       CHECK:   vector.transfer_write
//       CHECK:   scf.if
//       CHECK:     vector.transfer_read
//       CHECK:   } else {
//       CHECK:     vector.transfer_write
//       CHECK:   }
//       CHECK:   vector.transfer_write
//       CHECK:   return
func.func @forward_nested_negative(%arg0: i1, %arg1 : memref<4x4xf32>,
  %v0 : vector<1x4xf32>, %v1 : vector<1x4xf32>, %i : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cf0 = arith.constant 0.0 : f32
  vector.transfer_write %v0, %arg1[%c1, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, memref<4x4xf32>
  %x = scf.if %arg0 -> (vector<1x4xf32>) {
    %0 = vector.transfer_read %arg1[%c1, %c0], %cf0 {in_bounds = [true, true]} :
      memref<4x4xf32>, vector<1x4xf32>
    scf.yield %0 : vector<1x4xf32>
  } else {
    vector.transfer_write %v1, %arg1[%i, %c0] {in_bounds = [true, true]} :
      vector<1x4xf32>, memref<4x4xf32>
    scf.yield %v1 : vector<1x4xf32>
  }
  vector.transfer_write %x, %arg1[%c0, %i] {in_bounds = [true, true]} :
    vector<1x4xf32>, memref<4x4xf32>
  return
}

// CHECK-LABEL: func @dead_store_region
//       CHECK:   vector.transfer_write
//       CHECK:   scf.if
//       CHECK:   } else {
//       CHECK:     vector.transfer_read
//       CHECK:   }
//       CHECK:   scf.if
//   CHECK-NOT:     vector.transfer_write
//       CHECK:   }
//       CHECK:   vector.transfer_write
//   CHECK-NOT:   vector.transfer_write
//       CHECK:   vector.transfer_read
//       CHECK:   return
func.func @dead_store_region(%arg0: i1, %arg1 : memref<4x4xf32>,
  %v0 : vector<1x4xf32>, %v1 : vector<1x4xf32>, %i : index)
  -> (vector<1x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cf0 = arith.constant 0.0 : f32
  vector.transfer_write %v0, %arg1[%c1, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, memref<4x4xf32>
  %x = scf.if %arg0 -> (vector<1x4xf32>) {
    scf.yield %v1 : vector<1x4xf32>
  } else {
    %0 = vector.transfer_read %arg1[%i, %c0], %cf0 {in_bounds = [true, true]} :
      memref<4x4xf32>, vector<1x4xf32>
    scf.yield %0 : vector<1x4xf32>
  }
  scf.if %arg0 {
    vector.transfer_write %v0, %arg1[%c1, %c0] {in_bounds = [true, true]} :
      vector<1x4xf32>, memref<4x4xf32>
  }
  vector.transfer_write %x, %arg1[%c1, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, memref<4x4xf32>
  vector.transfer_write %x, %arg1[%c1, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, memref<4x4xf32>
  %1 = vector.transfer_read %arg1[%i, %c0], %cf0 {in_bounds = [true, true]} :
    memref<4x4xf32>, vector<1x4xf32>
  return %1 : vector<1x4xf32>
}

// CHECK-LABEL: func @dead_store_negative
//       CHECK:   scf.if
//       CHECK:     vector.transfer_write
//       CHECK:     vector.transfer_read
//       CHECK:   } else {
//       CHECK:   }
//       CHECK:   vector.transfer_write
//       CHECK:   return
func.func @dead_store_negative(%arg0: i1, %arg1 : memref<4x4xf32>,
  %v0 :vector<1x4xf32>, %v1 : vector<1x4xf32>, %i : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cf0 = arith.constant 0.0 : f32
  %x = scf.if %arg0 -> (vector<1x4xf32>) {
    vector.transfer_write %v0, %arg1[%c1, %c0] {in_bounds = [true, true]} :
      vector<1x4xf32>, memref<4x4xf32>
    %0 = vector.transfer_read %arg1[%i, %c0], %cf0 {in_bounds = [true, true]} :
      memref<4x4xf32>, vector<1x4xf32>
    scf.yield %0 : vector<1x4xf32>
  } else {
    scf.yield %v1 : vector<1x4xf32>
  }
  vector.transfer_write %x, %arg1[%c1, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, memref<4x4xf32>
  return
}

// CHECK-LABEL: func @dead_store_nested_region
//       CHECK:   scf.if
//       CHECK:     vector.transfer_read
//       CHECK:     scf.if
//   CHECK-NOT:       vector.transfer_write
//       CHECK:     }
//       CHECK:     vector.transfer_write
//       CHECK:   }
//       CHECK:   return
func.func @dead_store_nested_region(%arg0: i1, %arg1: i1, %arg2 : memref<4x4xf32>,
  %v0 : vector<1x4xf32>, %v1 : vector<1x4xf32>, %i : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cf0 = arith.constant 0.0 : f32
  scf.if %arg0 {
    %0 = vector.transfer_read %arg2[%i, %c0], %cf0 {in_bounds = [true, true]} :
      memref<4x4xf32>, vector<1x4xf32>
    scf.if %arg1 {
      vector.transfer_write %v1, %arg2[%c1, %c0] {in_bounds = [true, true]} :
        vector<1x4xf32>, memref<4x4xf32>
    }
    vector.transfer_write %v0, %arg2[%c1, %c0] {in_bounds = [true, true]} :
      vector<1x4xf32>, memref<4x4xf32>
  }
  return
}

// CHECK-LABEL: func @forward_dead_store_negative
//       CHECK:   vector.transfer_write
//       CHECK:   vector.transfer_write
//       CHECK:   vector.transfer_write
//       CHECK:   vector.transfer_write
//       CHECK:   vector.transfer_read
//       CHECK:   vector.transfer_write
//       CHECK:   return
func.func @forward_dead_store_negative(%arg0: i1, %arg1 : memref<4x4xf32>,
  %v0 : vector<1x4xf32>, %v1 : vector<1x1xf32>, %v2 : vector<1x4xf32>, %i : index) -> vector<1x4xf32> {
  %alias = memref.subview %arg1[0, 0] [2, 2] [1, 1] :
    memref<4x4xf32> to memref<2x2xf32, strided<[4, 1]>>
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  vector.transfer_write %v0, %arg1[%c1, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, memref<4x4xf32>
  // blocking write.
  vector.transfer_write %v1, %alias[%c0, %c0] {in_bounds = [true, true]} :
    vector<1x1xf32>, memref<2x2xf32, strided<[4, 1]>>
  vector.transfer_write %v2, %arg1[%c1, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, memref<4x4xf32>
  // blocking write.
  vector.transfer_write %v1, %alias[%c1, %c0] {in_bounds = [true, true]} :
    vector<1x1xf32>, memref<2x2xf32, strided<[4, 1]>>
  %0 = vector.transfer_read %arg1[%c1, %c0], %cf0 {in_bounds = [true, true]} :
    memref<4x4xf32>, vector<1x4xf32>
  vector.transfer_write %v2, %arg1[%c1, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, memref<4x4xf32>
  return %0 : vector<1x4xf32>
}


// Regression test - the following _potential forwarding_ of %1 to the final
// `vector.transfer_write` would not be safe:
//         %1 = vector.transfer_read %subview
//         vector.transfer_write %1, %alloca
//         vector.transfer_write %vec, %collapse_shape
//         %2 = vector.transfer_read %alloca
//         vector.transfer_write %1, %subview
// Indeed, %alloca and %collapse_shape alias and hence %2 != %1. Instead, the
// final `vector.transfer_write` should be preserved as:
//         vector.transfer_write %2, %subview

// CHECK-LABEL:  func.func @collapse_shape_and_read_from_source
//       CHECK:    scf.for {{.*}} {
//       CHECK:      vector.transfer_read
//       CHECK:      vector.transfer_write
//       CHECK:      vector.transfer_write
//       CHECK:      vector.transfer_read
//       CHECK:      vector.transfer_write

func.func @collapse_shape_and_read_from_source(%in_0: memref<1x20x1xi32>, %vec: vector<4xi32>) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c20 = arith.constant 20 : index

  %alloca = memref.alloca() {alignment = 64 : i64} : memref<1x4x1xi32>
  %collapse_shape = memref.collapse_shape %alloca [[0, 1, 2]] : memref<1x4x1xi32> into memref<4xi32>
  scf.for %arg0 = %c0 to %c20 step %c4 {
    %subview = memref.subview %in_0[0, %arg0, 0] [1, 4, 1] [1, 1, 1] : memref<1x20x1xi32> to memref<1x4x1xi32, strided<[20, 1, 1], offset: ?>>
    %1 = vector.transfer_read %subview[%c0, %c0, %c0], %c0_i32 {in_bounds = [true, true, true]} : memref<1x4x1xi32, strided<[20, 1, 1], offset: ?>>, vector<1x4x1xi32>
    // $alloca and $collapse_shape alias
    vector.transfer_write %1, %alloca[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x4x1xi32>, memref<1x4x1xi32>
    vector.transfer_write %vec, %collapse_shape[%c0] {in_bounds = [true]} : vector<4xi32>, memref<4xi32>
    %2 = vector.transfer_read %alloca[%c0, %c0, %c0], %c0_i32 {in_bounds = [true, true, true]} : memref<1x4x1xi32>, vector<1x4x1xi32>
    vector.transfer_write %2, %subview[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x4x1xi32>, memref<1x4x1xi32, strided<[20, 1, 1], offset: ?>>
  }
  return
}

// The same regression test for expand_shape.

// CHECK-LABEL:  func.func @expand_shape_and_read_from_source
//       CHECK:    scf.for {{.*}} {
//       CHECK:      vector.transfer_read
//       CHECK:      vector.transfer_write
//       CHECK:      vector.transfer_write
//       CHECK:      vector.transfer_read
//       CHECK:      vector.transfer_write

func.func @expand_shape_and_read_from_source(%in_0: memref<20xi32>, %vec: vector<1x4x1xi32>) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c20 = arith.constant 20 : index

  %alloca = memref.alloca() {alignment = 64 : i64} : memref<4xi32>
  %expand_shape = memref.expand_shape %alloca [[0, 1, 2]] output_shape [1, 4, 1] : memref<4xi32> into memref<1x4x1xi32>
  scf.for %arg0 = %c0 to %c20 step %c4 {
    %subview = memref.subview %in_0[%arg0] [4] [1] : memref<20xi32> to memref<4xi32, strided<[1], offset: ?>>
    %1 = vector.transfer_read %subview[%c0], %c0_i32 {in_bounds = [true]} : memref<4xi32, strided<[1], offset: ?>>, vector<4xi32>
    // $alloca and $expand_shape alias
    vector.transfer_write %1, %alloca[%c0] {in_bounds = [true]} : vector<4xi32>, memref<4xi32>
    vector.transfer_write %vec, %expand_shape[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x4x1xi32>, memref<1x4x1xi32>
    %2 = vector.transfer_read %alloca[%c0], %c0_i32 {in_bounds = [true]} : memref<4xi32>, vector<4xi32>
    vector.transfer_write %2, %subview[%c0] {in_bounds = [true]} : vector<4xi32>, memref<4xi32, strided<[1], offset: ?>>
  }
  return
}

// The same regression test, but the initial write is to the collapsed memref,
// and the subsequent unforwardable read is from the collapse shape.

// CHECK-LABEL:  func.func @collapse_shape_and_read_from_collapse
//       CHECK:    scf.for {{.*}} {
//       CHECK:      vector.transfer_read
//       CHECK:      vector.transfer_write
//       CHECK:      vector.transfer_write
//       CHECK:      vector.transfer_read
//       CHECK:      vector.transfer_write

func.func @collapse_shape_and_read_from_collapse(%in_0: memref<20xi32>, %vec: vector<1x4x1xi32>) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c20 = arith.constant 20 : index

  %alloca = memref.alloca() {alignment = 64 : i64} : memref<1x4x1xi32>
  %collapse_shape = memref.collapse_shape %alloca [[0, 1, 2]] : memref<1x4x1xi32> into memref<4xi32>
  scf.for %arg0 = %c0 to %c20 step %c4 {
    %subview = memref.subview %in_0[%arg0] [4] [1] : memref<20xi32> to memref<4xi32, strided<[1], offset: ?>>
    %1 = vector.transfer_read %subview[%c0], %c0_i32 {in_bounds = [true]} : memref<4xi32, strided<[1], offset: ?>>, vector<4xi32>
    vector.transfer_write %1, %collapse_shape[%c0] {in_bounds = [true]} : vector<4xi32>, memref<4xi32>
    // $alloca and $collapse_shape alias
    vector.transfer_write %vec, %alloca[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x4x1xi32>, memref<1x4x1xi32>
    %2 = vector.transfer_read %collapse_shape[%c0], %c0_i32 {in_bounds = [true]} : memref<4xi32>, vector<4xi32>
    vector.transfer_write %2, %subview[%c0] {in_bounds = [true]} : vector<4xi32>, memref<4xi32, strided<[1], offset: ?>>
  }
  return
}

// The same test except writing to the expanded source first (same as the
// previous collapse test but for expand).

// CHECK-LABEL:  func.func @expand_shape_and_read_from_expand
//       CHECK:    scf.for {{.*}} {
//       CHECK:      vector.transfer_read
//       CHECK:      vector.transfer_write
//       CHECK:      vector.transfer_write
//       CHECK:      vector.transfer_read
//       CHECK:      vector.transfer_write

func.func @expand_shape_and_read_from_expand(%in_0: memref<1x20x1xi32>, %vec: vector<4xi32>) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c20 = arith.constant 20 : index

  %alloca = memref.alloca() {alignment = 64 : i64} : memref<4xi32>
  %expand_shape = memref.expand_shape %alloca [[0, 1, 2]] output_shape [1, 4, 1] : memref<4xi32> into memref<1x4x1xi32>
  scf.for %arg0 = %c0 to %c20 step %c4 {
    %subview = memref.subview %in_0[0, %arg0, 0] [1, 4, 1] [1, 1, 1] : memref<1x20x1xi32> to memref<1x4x1xi32, strided<[20, 1, 1], offset: ?>>
    %1 = vector.transfer_read %subview[%c0, %c0, %c0], %c0_i32 {in_bounds = [true, true, true]} : memref<1x4x1xi32, strided<[20, 1, 1], offset: ?>>, vector<1x4x1xi32>
    vector.transfer_write %1, %expand_shape[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x4x1xi32>, memref<1x4x1xi32>
    // $alloca and $expand_shape alias
    vector.transfer_write %vec, %alloca[%c0] {in_bounds = [true]} : vector<4xi32>, memref<4xi32>
    %2 = vector.transfer_read %expand_shape[%c0, %c0, %c0], %c0_i32 {in_bounds = [true, true, true]} : memref<1x4x1xi32>, vector<1x4x1xi32>
    vector.transfer_write %2, %subview[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x4x1xi32>, memref<1x4x1xi32, strided<[20, 1, 1], offset: ?>>
  }
  return
}

// CHECK-LABEL: func @forward_dead_store_dynamic_same_index
//   CHECK-NOT:   vector.transfer_write
//   CHECK-NOT:   vector.transfer_read
//       CHECK:   scf.for
//       CHECK:   }
//       CHECK:   vector.transfer_write
//       CHECK:   return
func.func @forward_dead_store_dynamic_same_index(
    %buffer : memref<?x?xf32>, %v0 : vector<4xf32>, %v1 : vector<4xf32>, %i : index) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  vector.transfer_write %v0, %buffer[%i, %i] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32>
  // The following transfer op reads/writes to the same address so that we can forward.
  %0 = vector.transfer_read %buffer[%i, %i], %cf0 {in_bounds = [true]} : memref<?x?xf32>, vector<4xf32>
  %x = scf.for %i0 = %c0 to %c4 step %c1 iter_args(%acc = %0) -> (vector<4xf32>) {
    %1 = arith.addf %acc, %acc : vector<4xf32>
    scf.yield %1 : vector<4xf32>
  }
  vector.transfer_write %x, %buffer[%i, %i] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32>
  return
}

//   CHECK-LABEL: func @dont_forward_dead_store_dynamic_overlap
// CHECK-COUNT-2:   vector.transfer_write
//         CHECK:   vector.transfer_read
//         CHECK:   scf.for
//         CHECK:   }
//         CHECK:   vector.transfer_write
//         CHECK:   return
func.func @dont_forward_dead_store_dynamic_overlap(
    %buffer : memref<?x?xf32>, %v0 : vector<4xf32>, %v1 : vector<4xf32>, %i0 : index) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %i1 = affine.apply affine_map<(d0) -> (d0 + 3)>(%i0)
  vector.transfer_write %v0, %buffer[%i0, %i0] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32>
  // The following transfer op writes to an overlapping range so we cannot forward.
  vector.transfer_write %v0, %buffer[%i0, %i1] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32>
  %0 = vector.transfer_read %buffer[%i0, %i0], %cf0 {in_bounds = [true]} : memref<?x?xf32>, vector<4xf32>
  %x = scf.for %iv = %c0 to %c4 step %c1 iter_args(%acc = %0) -> (vector<4xf32>) {
    %1 = arith.addf %acc, %acc : vector<4xf32>
    scf.yield %1 : vector<4xf32>
  }
  vector.transfer_write %x, %buffer[%i0, %i0] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32>
  return
}

// CHECK-LABEL: func @forward_dead_store_dynamic_non_overlap_leading_dim
//       CHECK:   vector.transfer_write
//   CHECK-NOT:   vector.transfer_write
//   CHECK-NOT:   vector.transfer_read
//       CHECK:   scf.for
//       CHECK:   }
//       CHECK:   vector.transfer_write
//       CHECK:   return
func.func @forward_dead_store_dynamic_non_overlap_leading_dim(
    %buffer : memref<?x?xf32>, %v0 : vector<4xf32>, %v1 : vector<4xf32>, %i0 : index) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %i1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%i0)
  vector.transfer_write %v0, %buffer[%i0, %i0] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32>
  // The following transfer op writes to an non-overlapping range so we can forward.
  vector.transfer_write %v0, %buffer[%i1, %i0] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32>
  %0 = vector.transfer_read %buffer[%i0, %i0], %cf0 {in_bounds = [true]} : memref<?x?xf32>, vector<4xf32>
  %x = scf.for %iv = %c0 to %c4 step %c1 iter_args(%acc = %0) -> (vector<4xf32>) {
    %1 = arith.addf %acc, %acc : vector<4xf32>
    scf.yield %1 : vector<4xf32>
  }
  vector.transfer_write %x, %buffer[%i0, %i0] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32>
  return
}

// CHECK-LABEL: func @forward_dead_store_dynamic_non_overlap_trailing_dim
//       CHECK:   vector.transfer_write
//   CHECK-NOT:   vector.transfer_write
//   CHECK-NOT:   vector.transfer_read
//       CHECK:   scf.for
//       CHECK:   }
//       CHECK:   vector.transfer_write
//       CHECK:   return
func.func @forward_dead_store_dynamic_non_overlap_trailing_dim(
    %buffer : memref<?x?xf32>, %v0 : vector<4xf32>, %v1 : vector<4xf32>, %i0 : index) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %i1 = affine.apply affine_map<(d0) -> (d0 + 4)>(%i0)
  vector.transfer_write %v0, %buffer[%i0, %i0] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32>
  // The following transfer op writes to an non-overlapping range so we can forward.
  vector.transfer_write %v0, %buffer[%i0, %i1] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32>
  %0 = vector.transfer_read %buffer[%i0, %i0], %cf0 {in_bounds = [true]} : memref<?x?xf32>, vector<4xf32>
  %x = scf.for %iv = %c0 to %c4 step %c1 iter_args(%acc = %0) -> (vector<4xf32>) {
    %1 = arith.addf %acc, %acc : vector<4xf32>
    scf.yield %1 : vector<4xf32>
  }
  vector.transfer_write %x, %buffer[%i0, %i0] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32>
  return
}

// CHECK-LABEL: func @forward_dead_constant_splat_store_with_masking
//       CHECK:   %[[SPLAT:.*]] = arith.constant dense<0.000000e+00> : vector<[8]x[8]xf32>
//   CHECK-NOT:   vector.transfer_write
//   CHECK-NOT:   vector.transfer_read
//       CHECK:   scf.for
//  CHECK-SAME:     iter_args(%{{.*}} = %[[SPLAT]])
//       CHECK:   }
//       CHECK:   vector.transfer_write
//       CHECK:   return
func.func @forward_dead_constant_splat_store_with_masking(%buffer : memref<?x?xf32>, %mask: vector<[8]x[8]xi1>) {
  %zero_splat = arith.constant dense<0.0> : vector<[8]x[8]xf32>
  %read_padding = arith.constant 0.0 : f32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  vector.transfer_write %zero_splat, %buffer[%c0, %c0], %mask {in_bounds = [true, true]} : vector<[8]x[8]xf32>, memref<?x?xf32>
  %0 = vector.transfer_read %buffer[%c0, %c0], %read_padding, %mask {in_bounds = [true, true]} : memref<?x?xf32>, vector<[8]x[8]xf32>
  %x = scf.for %arg2 = %c0 to %c512 step %c1 iter_args(%acc = %0) -> (vector<[8]x[8]xf32>) {
    %1 = arith.addf %acc, %acc : vector<[8]x[8]xf32>
    scf.yield %1 : vector<[8]x[8]xf32>
  }
  vector.transfer_write %x, %buffer[%c0, %c0], %mask {in_bounds = [true, true]} : vector<[8]x[8]xf32>, memref<?x?xf32>
  return
}

// Here the read can be eliminated but not the write (due to mismatched masks).
// CHECK-LABEL: func @forward_dead_constant_splat_store_with_masking_unmasked_write
//       CHECK:   %[[SPLAT:.*]] = arith.constant dense<0.000000e+00> : vector<[8]x[8]xf32>
//       CHECK:   vector.transfer_write %[[SPLAT]]
//       CHECK:   scf.for
//  CHECK-SAME:     iter_args(%{{.*}} = %[[SPLAT]])
//       CHECK:   }
//       CHECK:   vector.transfer_write
//       CHECK:   return
func.func @forward_dead_constant_splat_store_with_masking_unmasked_write(%buffer : memref<?x?xf32>, %mask: vector<[8]x[8]xi1>) {
  %zero_splat = arith.constant dense<0.0> : vector<[8]x[8]xf32>
  %read_padding = arith.constant 0.0 : f32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  vector.transfer_write %zero_splat, %buffer[%c0, %c0] {in_bounds = [true, true]} : vector<[8]x[8]xf32>, memref<?x?xf32>
  %0 = vector.transfer_read %buffer[%c0, %c0], %read_padding, %mask {in_bounds = [true, true]} : memref<?x?xf32>, vector<[8]x[8]xf32>
  %x = scf.for %arg2 = %c0 to %c512 step %c1 iter_args(%acc = %0) -> (vector<[8]x[8]xf32>) {
    %1 = arith.addf %acc, %acc : vector<[8]x[8]xf32>
    scf.yield %1 : vector<[8]x[8]xf32>
  }
  vector.transfer_write %x, %buffer[%c0, %c0], %mask {in_bounds = [true, true]} : vector<[8]x[8]xf32>, memref<?x?xf32>
  return
}

// Negative test, the padding does not match the constant splat, so we can't
// forward the store.
// CHECK-LABEL: func @forward_dead_constant_splat_store_with_masking_negative_0
//       CHECK:   vector.transfer_write
//       CHECK:   vector.transfer_read
//       CHECK:   scf.for
//       CHECK:   }
//       CHECK:   vector.transfer_write
//       CHECK:   return
func.func @forward_dead_constant_splat_store_with_masking_negative_0(%buffer : memref<?x?xf32>, %mask: vector<[8]x[8]xi1>) {
  %zero_splat = arith.constant dense<0.0> : vector<[8]x[8]xf32>
  %read_padding = arith.constant 1.0 : f32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  vector.transfer_write %zero_splat, %buffer[%c0, %c0], %mask {in_bounds = [true, true]} : vector<[8]x[8]xf32>, memref<?x?xf32>
  %0 = vector.transfer_read %buffer[%c0, %c0], %read_padding, %mask {in_bounds = [true, true]} : memref<?x?xf32>, vector<[8]x[8]xf32>
  %x = scf.for %arg2 = %c0 to %c512 step %c1 iter_args(%acc = %0) -> (vector<[8]x[8]xf32>) {
    %1 = arith.addf %acc, %acc : vector<[8]x[8]xf32>
    scf.yield %1 : vector<[8]x[8]xf32>
  }
  vector.transfer_write %x, %buffer[%c0, %c0], %mask {in_bounds = [true, true]} : vector<[8]x[8]xf32>, memref<?x?xf32>
  return
}

// Negative test, the masks don't match between the read and write, so we can't
// foward the store.
// CHECK-LABEL: func @forward_dead_constant_splat_store_with_masking_negative_1
//       CHECK:   vector.transfer_write
//       CHECK:   vector.transfer_read
//       CHECK:   scf.for
//       CHECK:   }
//       CHECK:   vector.transfer_write
//       CHECK:   return
func.func @forward_dead_constant_splat_store_with_masking_negative_1(%buffer : memref<?x?xf32>, %mask_a: vector<[8]x[8]xi1>, %mask_b: vector<[8]x[8]xi1>) {
  %zero_splat = arith.constant dense<0.0> : vector<[8]x[8]xf32>
  %read_padding = arith.constant 0.0 : f32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  vector.transfer_write %zero_splat, %buffer[%c0, %c0], %mask_a {in_bounds = [true, true]} : vector<[8]x[8]xf32>, memref<?x?xf32>
  %0 = vector.transfer_read %buffer[%c0, %c0], %read_padding, %mask_b {in_bounds = [true, true]} : memref<?x?xf32>, vector<[8]x[8]xf32>
  %x = scf.for %arg2 = %c0 to %c512 step %c1 iter_args(%acc = %0) -> (vector<[8]x[8]xf32>) {
    %1 = arith.addf %acc, %acc : vector<[8]x[8]xf32>
    scf.yield %1 : vector<[8]x[8]xf32>
  }
  vector.transfer_write %x, %buffer[%c0, %c0], %mask_a {in_bounds = [true, true]} : vector<[8]x[8]xf32>, memref<?x?xf32>
  return
}

// Negative test, here the write is masked but the read is unmasked. We can't
// forward the store (as the write could be of less elements then the read).
// CHECK-LABEL: func @forward_dead_constant_splat_store_with_masking_negative_3
//       CHECK:   vector.transfer_write
//       CHECK:   vector.transfer_read
//       CHECK:   scf.for
//       CHECK:   }
//       CHECK:   vector.transfer_write
//       CHECK:   return
func.func @forward_dead_constant_splat_store_with_masking_negative_3(%buffer : memref<?x?xf32>, %mask: vector<[8]x[8]xi1>) {
  %zero_splat = arith.constant dense<0.0> : vector<[8]x[8]xf32>
  %read_padding = arith.constant 0.0 : f32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  vector.transfer_write %zero_splat, %buffer[%c0, %c0], %mask {in_bounds = [true, true]} : vector<[8]x[8]xf32>, memref<?x?xf32>
  %0 = vector.transfer_read %buffer[%c0, %c0], %read_padding {in_bounds = [true, true]} : memref<?x?xf32>, vector<[8]x[8]xf32>
  %x = scf.for %arg2 = %c0 to %c512 step %c1 iter_args(%acc = %0) -> (vector<[8]x[8]xf32>) {
    %1 = arith.addf %acc, %acc : vector<[8]x[8]xf32>
    scf.yield %1 : vector<[8]x[8]xf32>
  }
  vector.transfer_write %x, %buffer[%c0, %c0], %mask {in_bounds = [true, true]} : vector<[8]x[8]xf32>, memref<?x?xf32>
  return
}

// Here each read/write is to a different subview, but they all point to exact
// same bit of memory (just through casts and subviews with unit strides and
// zero offsets).
// CHECK-LABEL: func @forward_and_eliminate_stores_through_trivial_aliases
//   CHECK-NOT:   vector.transfer_write
//   CHECK-NOT:   vector.transfer_read
//       CHECK:   scf.for
//       CHECK:   }
//       CHECK:   vector.transfer_write
//       CHECK:   return
func.func @forward_and_eliminate_stores_through_trivial_aliases(
  %buffer : memref<?x?xf32>, %vec: vector<[8]x[8]xf32>, %size: index, %a_size: index, %another_size: index
) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %cst = arith.constant 0.0 : f32
  vector.transfer_write %vec, %buffer[%c0, %c0] {in_bounds = [true, true]} : vector<[8]x[8]xf32>, memref<?x?xf32>
  %direct_subview = memref.subview %buffer[0, 0] [%a_size, %a_size] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1]>>
  %cast = memref.cast %direct_subview : memref<?x?xf32, strided<[?, 1]>> to memref<?x?xf32>
  %subview_of_cast = memref.subview %cast[0, 0] [%another_size, %another_size] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1]>>
  %21 = vector.transfer_read %direct_subview[%c0, %c0], %cst {in_bounds = [true, true]} : memref<?x?xf32, strided<[?, 1]>>, vector<[8]x[8]xf32>
  %23 = scf.for %arg2 = %c0 to %c32 step %c1 iter_args(%arg3 = %21) -> (vector<[8]x[8]xf32>) {
    %24 = arith.addf %arg3, %arg3 : vector<[8]x[8]xf32>
    scf.yield %24 : vector<[8]x[8]xf32>
  }
  vector.transfer_write %23, %subview_of_cast[%c0, %c0] {in_bounds = [true, true]} : vector<[8]x[8]xf32>, memref<?x?xf32, strided<[?, 1]>>
  return
}
