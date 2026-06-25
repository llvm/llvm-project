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

// -----

// Vector delinearize: unrolled to per-element affine.apply.

//   CHECK-DAG:   #[[$DIV8:.+]] = affine_map<()[s0] -> (s0 floordiv 8)>
//   CHECK-DAG:   #[[$MOD8:.+]] = affine_map<()[s0] -> (s0 mod 8)>

// CHECK-LABEL: @delinearize_vector_unroll
// CHECK-SAME:    (%[[VEC:.+]]: vector<2xindex>)
// CHECK:         %[[POISON0:.+]] = ub.poison : vector<2xindex>
// CHECK:         %[[S0:.+]] = vector.extract %[[VEC]][0]
// CHECK:         %[[D0:.+]] = affine.apply #[[$DIV8]]()[%[[S0]]]
// CHECK:         %[[M0:.+]] = affine.apply #[[$MOD8]]()[%[[S0]]]
// CHECK:         %[[R0_0:.+]] = vector.insert %[[D0]], %[[POISON0]] [0]
// CHECK:         %[[R1_0:.+]] = vector.insert %[[M0]], %[[POISON0]] [0]
// CHECK:         %[[S1:.+]] = vector.extract %[[VEC]][1]
// CHECK:         %[[D1:.+]] = affine.apply #[[$DIV8]]()[%[[S1]]]
// CHECK:         %[[M1:.+]] = affine.apply #[[$MOD8]]()[%[[S1]]]
// CHECK:         %[[R0_1:.+]] = vector.insert %[[D1]], %[[R0_0]] [1]
// CHECK:         %[[R1_1:.+]] = vector.insert %[[M1]], %[[R1_0]] [1]
// CHECK:         return %[[R0_1]], %[[R1_1]]
func.func @delinearize_vector_unroll(%vec: vector<2xindex>) -> (vector<2xindex>, vector<2xindex>) {
  %0:2 = affine.delinearize_index %vec into (4, 8) : vector<2xindex>, vector<2xindex>
  return %0#0, %0#1 : vector<2xindex>, vector<2xindex>
}

// -----

// Vector linearize: unrolled to per-element affine.apply.

// CHECK-DAG:   #[[$LIN:.+]] = affine_map<()[s0, s1] -> (s0 * 8 + s1)>

// CHECK-LABEL: @linearize_vector_unroll
// CHECK-SAME:    (%[[V0:.+]]: vector<2xindex>, %[[V1:.+]]: vector<2xindex>)
// CHECK:         %[[POISON:.+]] = ub.poison : vector<2xindex>
// CHECK:         %[[A0:.+]] = vector.extract %[[V0]][0]
// CHECK:         %[[B0:.+]] = vector.extract %[[V1]][0]
// CHECK:         %[[L0:.+]] = affine.apply #[[$LIN]]()[%[[A0]], %[[B0]]]
// CHECK:         %[[R0:.+]] = vector.insert %[[L0]], %[[POISON]] [0]
// CHECK:         %[[A1:.+]] = vector.extract %[[V0]][1]
// CHECK:         %[[B1:.+]] = vector.extract %[[V1]][1]
// CHECK:         %[[L1:.+]] = affine.apply #[[$LIN]]()[%[[A1]], %[[B1]]]
// CHECK:         %[[R1:.+]] = vector.insert %[[L1]], %[[R0]] [1]
// CHECK:         return %[[R1]]
func.func @linearize_vector_unroll(%v0: vector<2xindex>, %v1: vector<2xindex>) -> vector<2xindex> {
  %0 = affine.linearize_index [%v0, %v1] by (4, 8) : vector<2xindex>
  return %0 : vector<2xindex>
}

// -----

// Multi-dimensional vector delinearize: unrolled with static positions.

//   CHECK-DAG:   #[[$DIV8:.+]] = affine_map<()[s0] -> (s0 floordiv 8)>
//   CHECK-DAG:   #[[$MOD8:.+]] = affine_map<()[s0] -> (s0 mod 8)>

// CHECK-LABEL: @delinearize_2d_vector_unroll
// CHECK-SAME:    (%[[VEC:.+]]: vector<2x2xindex>)
// CHECK:         %[[POISON:.+]] = ub.poison : vector<2x2xindex>
// CHECK:         %[[S00:.+]] = vector.extract %[[VEC]][0, 0]
// CHECK:         %[[D00:.+]] = affine.apply #[[$DIV8]]()[%[[S00]]]
// CHECK:         %[[M00:.+]] = affine.apply #[[$MOD8]]()[%[[S00]]]
// CHECK:         %[[R0_00:.+]] = vector.insert %[[D00]], %[[POISON]] [0, 0]
// CHECK:         %[[R1_00:.+]] = vector.insert %[[M00]], %[[POISON]] [0, 0]
// CHECK:         %[[S01:.+]] = vector.extract %[[VEC]][0, 1]
// CHECK:         %[[D01:.+]] = affine.apply #[[$DIV8]]()[%[[S01]]]
// CHECK:         %[[M01:.+]] = affine.apply #[[$MOD8]]()[%[[S01]]]
// CHECK:         vector.insert %[[D01]], %[[R0_00]] [0, 1]
// CHECK:         vector.insert %[[M01]], %[[R1_00]] [0, 1]
// CHECK:         vector.extract %[[VEC]][1, 0]
// CHECK:         vector.extract %[[VEC]][1, 1]
func.func @delinearize_2d_vector_unroll(%vec: vector<2x2xindex>) -> (vector<2x2xindex>, vector<2x2xindex>) {
  %0:2 = affine.delinearize_index %vec into (4, 8) : vector<2x2xindex>, vector<2x2xindex>
  return %0#0, %0#1 : vector<2x2xindex>, vector<2x2xindex>
}
