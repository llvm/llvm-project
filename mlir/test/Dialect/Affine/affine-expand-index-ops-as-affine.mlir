// RUN: mlir-opt %s -affine-expand-index-ops-as-affine -split-input-file | FileCheck %s

//   CHECK-DAG:   #[[$map0:.+]] = affine_map<()[s0] -> (s0 floordiv 50176)>
//   CHECK-DAG:   #[[$map1:.+]] = affine_map<()[s0] -> ((s0 mod 50176) floordiv 224)>
//   CHECK-DAG:   #[[$map2:.+]] = affine_map<()[s0] -> (s0 mod 224)>

// CHECK-LABEL: @static_basis
//  CHECK-SAME:    (%[[IDX:.+]]: index)
//       CHECK:   %[[N:.+]] = affine.apply #[[$map0]]()[%[[IDX]]]
//       CHECK:   %[[P:.+]] = affine.apply #[[$map1]]()[%[[IDX]]]
//       CHECK:   %[[Q:.+]] = affine.apply #[[$map2]]()[%[[IDX]]]
//       CHECK:   return %[[N]], %[[P]], %[[Q]]
func.func @static_basis(%linear_index: index) -> (index, index, index) {
  %1:3 = affine.delinearize_index %linear_index into (16, 224, 224) : index, index, index
  return %1#0, %1#1, %1#2 : index, index, index
}

// -----

//   CHECK-DAG:   #[[$map0:.+]] = affine_map<()[s0, s1, s2] -> (s2 floordiv (s0 * s1))>
//   CHECK-DAG:   #[[$map1:.+]] = affine_map<()[s0, s1, s2] -> ((s2 mod (s0 * s1)) floordiv s1)>
//   CHECK-DAG:   #[[$map2:.+]] = affine_map<()[s0, s1, s2] -> ((s2 mod (s0 * s1)) mod s1)>

// CHECK-LABEL: @dynamic_basis
//  CHECK-SAME:    (%[[IDX:.+]]: index, %[[MEMREF:.+]]: memref
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//        CHECK:  %[[DIM1:.+]] = memref.dim %[[MEMREF]], %[[C1]] :
//        CHECK:  %[[DIM2:.+]] = memref.dim %[[MEMREF]], %[[C2]] :
//       CHECK:   %[[N:.+]] = affine.apply #[[$map0]]()[%[[DIM1]], %[[DIM2]], %[[IDX]]]
//       CHECK:   %[[P:.+]] = affine.apply #[[$map1]]()[%[[DIM1]], %[[DIM2]], %[[IDX]]]
//       CHECK:   %[[Q:.+]] = affine.apply #[[$map2]]()[%[[DIM1]], %[[DIM2]], %[[IDX]]]
//       CHECK:   return %[[N]], %[[P]], %[[Q]]
func.func @dynamic_basis(%linear_index: index, %src: memref<?x?x?xf32>) -> (index, index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %b1 = memref.dim %src, %c1 : memref<?x?x?xf32>
  %b2 = memref.dim %src, %c2 : memref<?x?x?xf32>
  // Note: no outer bound.
  %1:3 = affine.delinearize_index %linear_index into (%b1, %b2) : index, index, index
  return %1#0, %1#1, %1#2 : index, index, index
}

// -----

// CHECK-DAG: #[[$map0:.+]] = affine_map<()[s0, s1, s2] -> (s0 * 15 + s1 * 5 + s2)>

// CHECK-LABEL: @linearize_static
// CHECK-SAME: (%[[arg0:.+]]: index, %[[arg1:.+]]: index, %[[arg2:.+]]: index)
// CHECK: %[[val_0:.+]] = affine.apply #[[$map0]]()[%[[arg0]], %[[arg1]], %[[arg2]]]
// CHECK: return %[[val_0]]
func.func @linearize_static(%arg0: index, %arg1: index, %arg2: index) -> index {
  %0 = affine.linearize_index [%arg0, %arg1, %arg2] by (2, 3, 5) : index
  func.return %0 : index
}

// -----

// CHECK-DAG: #[[$map0:.+]] =  affine_map<()[s0, s1, s2, s3, s4] -> (s1 * s2 + s3 + s0 * (s2 * s4))>

// CHECK-LABEL: @linearize_dynamic
// CHECK-SAME: (%[[arg0:.+]]: index, %[[arg1:.+]]: index, %[[arg2:.+]]: index, %[[arg3:.+]]: index, %[[arg4:.+]]: index)
// CHECK: %[[val_0:.+]] = affine.apply #[[$map0]]()[%[[arg0]], %[[arg1]], %[[arg4]], %[[arg2]], %[[arg3]]]
// CHECK: return %[[val_0]]
func.func @linearize_dynamic(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> index {
  // Note: no outer bounds
  %0 = affine.linearize_index [%arg0, %arg1, %arg2] by (%arg3, %arg4) : index
  func.return %0 : index
}
