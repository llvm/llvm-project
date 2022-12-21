// RUN: mlir-opt %s -test-scalar-vector-transfer-lowering -split-input-file | FileCheck %s

// CHECK-LABEL: func @transfer_read_0d(
//  CHECK-SAME:     %[[m:.*]]: memref<?x?x?xf32>, %[[idx:.*]]: index
//       CHECK:   %[[r:.*]] = memref.load %[[m]][%[[idx]], %[[idx]], %[[idx]]]
//       CHECK:   return %[[r]]
func.func @transfer_read_0d(%m: memref<?x?x?xf32>, %idx: index) -> f32 {
  %cst = arith.constant 0.0 : f32
  %0 = vector.transfer_read %m[%idx, %idx, %idx], %cst : memref<?x?x?xf32>, vector<f32>
  %1 = vector.extractelement %0[] : vector<f32>
  return %1 : f32
}

// -----

//       CHECK: #[[$map:.*]] = affine_map<()[s0, s1] -> (s0 + s1)>
// CHECK-LABEL: func @transfer_read_1d(
//  CHECK-SAME:     %[[m:.*]]: memref<?x?x?xf32>, %[[idx:.*]]: index, %[[idx2:.*]]: index
//       CHECK:   %[[added:.*]] = affine.apply #[[$map]]()[%[[idx]], %[[idx2]]]
//       CHECK:   %[[r:.*]] = memref.load %[[m]][%[[idx]], %[[idx]], %[[added]]]
//       CHECK:   return %[[r]]
func.func @transfer_read_1d(%m: memref<?x?x?xf32>, %idx: index, %idx2: index) -> f32 {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %0 = vector.transfer_read %m[%idx, %idx, %idx], %cst {in_bounds = [true]} : memref<?x?x?xf32>, vector<5xf32>
  %1 = vector.extractelement %0[%idx2 : index] : vector<5xf32>
  return %1 : f32
}

// -----

// CHECK-LABEL: func @tensor_transfer_read_0d(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x?x?xf32>, %[[idx:.*]]: index
//       CHECK:   %[[r:.*]] = tensor.extract %[[t]][%[[idx]], %[[idx]], %[[idx]]]
//       CHECK:   return %[[r]]
func.func @tensor_transfer_read_0d(%t: tensor<?x?x?xf32>, %idx: index) -> f32 {
  %cst = arith.constant 0.0 : f32
  %0 = vector.transfer_read %t[%idx, %idx, %idx], %cst : tensor<?x?x?xf32>, vector<f32>
  %1 = vector.extractelement %0[] : vector<f32>
  return %1 : f32
}

// -----

// CHECK-LABEL: func @transfer_write_0d(
//  CHECK-SAME:     %[[m:.*]]: memref<?x?x?xf32>, %[[idx:.*]]: index, %[[f:.*]]: f32
//       CHECK:   memref.store %[[f]], %[[m]][%[[idx]], %[[idx]], %[[idx]]]
func.func @transfer_write_0d(%m: memref<?x?x?xf32>, %idx: index, %f: f32) {
  %0 = vector.broadcast %f : f32 to vector<f32>
  vector.transfer_write %0, %m[%idx, %idx, %idx] : vector<f32>, memref<?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: func @transfer_write_1d(
//  CHECK-SAME:     %[[m:.*]]: memref<?x?x?xf32>, %[[idx:.*]]: index, %[[f:.*]]: f32
//       CHECK:   memref.store %[[f]], %[[m]][%[[idx]], %[[idx]], %[[idx]]]
func.func @transfer_write_1d(%m: memref<?x?x?xf32>, %idx: index, %f: f32) {
  %0 = vector.broadcast %f : f32 to vector<1xf32>
  vector.transfer_write %0, %m[%idx, %idx, %idx] : vector<1xf32>, memref<?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: func @tensor_transfer_write_0d(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x?x?xf32>, %[[idx:.*]]: index, %[[f:.*]]: f32
//       CHECK:   %[[r:.*]] = tensor.insert %[[f]] into %[[t]][%[[idx]], %[[idx]], %[[idx]]]
//       CHECK:   return %[[r]]
func.func @tensor_transfer_write_0d(%t: tensor<?x?x?xf32>, %idx: index, %f: f32) -> tensor<?x?x?xf32> {
  %0 = vector.broadcast %f : f32 to vector<f32>
  %1 = vector.transfer_write %0, %t[%idx, %idx, %idx] : vector<f32>, tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}
