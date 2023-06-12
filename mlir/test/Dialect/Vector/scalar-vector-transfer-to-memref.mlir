// RUN: mlir-opt %s -test-scalar-vector-transfer-lowering -split-input-file | FileCheck %s
// RUN: mlir-opt %s -test-scalar-vector-transfer-lowering=allow-multiple-uses -split-input-file | FileCheck %s --check-prefix=MULTIUSE

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
//       CHECK:   %[[bc:.*]] = vector.broadcast %[[f]] : f32 to vector<f32>
//       CHECK:   %[[extract:.*]] = vector.extractelement %[[bc]][] : vector<f32>
//       CHECK:   memref.store %[[extract]], %[[m]][%[[idx]], %[[idx]], %[[idx]]]
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
//       CHECK:   %[[bc:.*]] = vector.broadcast %[[f]] : f32 to vector<f32>
//       CHECK:   %[[extract:.*]] = vector.extractelement %[[bc]][] : vector<f32>
//       CHECK:   %[[r:.*]] = tensor.insert %[[extract]] into %[[t]][%[[idx]], %[[idx]], %[[idx]]]
//       CHECK:   return %[[r]]
func.func @tensor_transfer_write_0d(%t: tensor<?x?x?xf32>, %idx: index, %f: f32) -> tensor<?x?x?xf32> {
  %0 = vector.broadcast %f : f32 to vector<f32>
  %1 = vector.transfer_write %0, %t[%idx, %idx, %idx] : vector<f32>, tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}

// -----

//       CHECK: #[[$map:.*]] = affine_map<()[s0] -> (s0 + 8)>
//       CHECK: #[[$map1:.*]] = affine_map<()[s0] -> (s0 + 1)>
// CHECK-LABEL: func @transfer_read_2d_extract(
//  CHECK-SAME:     %[[m:.*]]: memref<?x?x?x?xf32>, %[[idx:.*]]: index, %[[idx2:.*]]: index
//       CHECK:   %[[added:.*]] = affine.apply #[[$map]]()[%[[idx]]]
//       CHECK:   %[[added1:.*]] = affine.apply #[[$map1]]()[%[[idx]]]
//       CHECK:   %[[r:.*]] = memref.load %[[m]][%[[idx]], %[[idx]], %[[added]], %[[added1]]]
//       CHECK:   return %[[r]]
func.func @transfer_read_2d_extract(%m: memref<?x?x?x?xf32>, %idx: index, %idx2: index) -> f32 {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %0 = vector.transfer_read %m[%idx, %idx, %idx, %idx], %cst {in_bounds = [true, true]} : memref<?x?x?x?xf32>, vector<10x5xf32>
  %1 = vector.extract %0[8, 1] : vector<10x5xf32>
  return %1 : f32
}

// -----

// CHECK-LABEL: func @transfer_write_arith_constant(
//  CHECK-SAME:     %[[m:.*]]: memref<?x?x?xf32>, %[[idx:.*]]: index
//       CHECK:   %[[cst:.*]] = arith.constant dense<5.000000e+00> : vector<1x1xf32>
//       CHECK:   %[[extract:.*]] = vector.extract %[[cst]][0, 0] : vector<1x1xf32>
//       CHECK:   memref.store %[[extract]], %[[m]][%[[idx]], %[[idx]], %[[idx]]]
func.func @transfer_write_arith_constant(%m: memref<?x?x?xf32>, %idx: index) {
  %cst = arith.constant dense<5.000000e+00> : vector<1x1xf32>
  vector.transfer_write %cst, %m[%idx, %idx, %idx] : vector<1x1xf32>, memref<?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: func @transfer_read_multi_use(
//  CHECK-SAME:   %[[m:.*]]: memref<?xf32>, %[[idx:.*]]: index
//   CHECK-NOT:   memref.load
//       CHECK:   %[[r:.*]] = vector.transfer_read %[[m]][%[[idx]]]
//       CHECK:   %[[e0:.*]] = vector.extract %[[r]][0]
//       CHECK:   %[[e1:.*]] = vector.extract %[[r]][1]
//       CHECK:   return %[[e0]], %[[e1]]

// MULTIUSE-LABEL: func @transfer_read_multi_use(
//  MULTIUSE-SAME:   %[[m:.*]]: memref<?xf32>, %[[idx0:.*]]: index
//   MULTIUSE-NOT:   vector.transfer_read
//       MULTIUSE:   %[[r0:.*]] = memref.load %[[m]][%[[idx0]]
//       MULTIUSE:   %[[idx1:.*]] = affine.apply
//       MULTIUSE:   %[[r1:.*]] = memref.load %[[m]][%[[idx1]]
//       MULTIUSE:   return %[[r0]], %[[r1]]

func.func @transfer_read_multi_use(%m: memref<?xf32>, %idx: index) -> (f32, f32) {
  %cst = arith.constant 0.0 : f32
  %0 = vector.transfer_read %m[%idx], %cst {in_bounds = [true]} : memref<?xf32>, vector<16xf32>
  %1 = vector.extract %0[0] : vector<16xf32>
  %2 = vector.extract %0[1] : vector<16xf32>
  return %1, %2 : f32, f32
}

// -----

// Check that patterns don't trigger for an sub-vector (not scalar) extraction.
// CHECK-LABEL: func @subvector_extract(
//  CHECK-SAME:   %[[m:.*]]: memref<?x?xf32>, %[[idx:.*]]: index
//   CHECK-NOT:   memref.load
//       CHECK:   %[[r:.*]] = vector.transfer_read %[[m]][%[[idx]], %[[idx]]]
//       CHECK:   %[[e0:.*]] = vector.extract %[[r]][0]
//       CHECK:   return %[[e0]]

func.func @subvector_extract(%m: memref<?x?xf32>, %idx: index) -> vector<16xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = vector.transfer_read %m[%idx, %idx], %cst {in_bounds = [true, true]} : memref<?x?xf32>, vector<8x16xf32>
  %1 = vector.extract %0[0] : vector<8x16xf32>
  return %1 : vector<16xf32>
}

