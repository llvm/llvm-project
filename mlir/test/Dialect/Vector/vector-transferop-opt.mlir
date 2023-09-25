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

// CHECK-LABEL:  func.func @collapse_shape
//       CHECK:    scf.for {{.*}} {
//       CHECK:      vector.transfer_read
//       CHECK:      vector.transfer_write
//       CHECK:      vector.transfer_write
//       CHECK:      vector.transfer_read
//       CHECK:      vector.transfer_write

func.func @collapse_shape(%in_0: memref<1x20x1xi32>, %vec: vector<4xi32>) {
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
