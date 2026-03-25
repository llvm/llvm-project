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

// CHECK-LABEL: @expand_delinearize_vector
// CHECK-SAME:    (%[[VEC:.+]]: vector<16xindex>)
// CHECK-DAG:     %[[C8:.+]] = arith.constant dense<8> : vector<16xindex>
// CHECK:         %[[DIV:.+]] = arith.divsi %[[VEC]], %[[C8]]
// CHECK:         %[[MUL:.+]] = arith.muli %[[DIV]], %[[C8]]
// CHECK:         %[[REM:.+]] = arith.subi %[[VEC]], %[[MUL]]
// CHECK:         return %[[DIV]], %[[REM]]
func.func @expand_delinearize_vector(%vec: vector<16xindex>) -> (vector<16xindex>, vector<16xindex>) {
  %0:2 = affine.delinearize_index %vec into (4, 8) : vector<16xindex>, vector<16xindex>
  return %0#0, %0#1 : vector<16xindex>, vector<16xindex>
}

// -----

// CHECK-LABEL: @expand_linearize_vector
// CHECK-SAME:    (%[[V0:.+]]: vector<16xindex>, %[[V1:.+]]: vector<16xindex>)
// CHECK-DAG:     %[[C8:.+]] = arith.constant dense<8> : vector<16xindex>
// CHECK:         %[[MUL:.+]] = arith.muli %[[V0]], %[[C8]]
// CHECK:         %[[ADD:.+]] = arith.addi %[[MUL]], %[[V1]]
// CHECK:         return %[[ADD]]
func.func @expand_linearize_vector(%v0: vector<16xindex>, %v1: vector<16xindex>) -> vector<16xindex> {
  %0 = affine.linearize_index [%v0, %v1] by (4, 8) : vector<16xindex>
  return %0 : vector<16xindex>
}

// -----

// CHECK-LABEL: @expand_delinearize_vector_3d
// CHECK-SAME:    (%[[VEC:.+]]: vector<16xindex>)
// CHECK-DAG:     %[[C4:.+]] = arith.constant dense<4> : vector<16xindex>
// CHECK-DAG:     %[[C12:.+]] = arith.constant dense<12> : vector<16xindex>
// CHECK:         %[[D0:.+]] = arith.divsi %[[VEC]], %[[C12]]
// CHECK:         %[[M0:.+]] = arith.muli %[[D0]], %[[C12]]
// CHECK:         %[[R0:.+]] = arith.subi %[[VEC]], %[[M0]]
// CHECK:         %[[D1:.+]] = arith.divsi %[[R0]], %[[C4]]
// CHECK:         %[[M1:.+]] = arith.muli %[[D1]], %[[C4]]
// CHECK:         %[[R1:.+]] = arith.subi %[[R0]], %[[M1]]
// CHECK:         return %[[D0]], %[[D1]], %[[R1]]
func.func @expand_delinearize_vector_3d(%vec: vector<16xindex>) -> (vector<16xindex>, vector<16xindex>, vector<16xindex>) {
  %0:3 = affine.delinearize_index %vec into (2, 3, 4) : vector<16xindex>, vector<16xindex>, vector<16xindex>
  return %0#0, %0#1, %0#2 : vector<16xindex>, vector<16xindex>, vector<16xindex>
}

// -----

// Vector linearize -> offset -> delinearize pattern
// (as would be used in vector.gather lowering).

// CHECK-LABEL: @vector_linearize_offset_delinearize
// CHECK-SAME:    (%[[V0:.+]]: vector<4xindex>, %[[V1:.+]]: vector<4xindex>, %[[OFF:.+]]: vector<4xindex>)
// CHECK-DAG:     %[[C8:.+]] = arith.constant dense<8> : vector<4xindex>
// CHECK:         %[[LIN:.+]] = arith.muli %[[V0]], %[[C8]]
// CHECK:         %[[LIN2:.+]] = arith.addi %[[LIN]], %[[V1]]
// CHECK:         %[[FLAT:.+]] = arith.addi %[[LIN2]], %[[OFF]]
// CHECK:         %[[DIV:.+]] = arith.divsi %[[FLAT]], %[[C8]]
// CHECK:         %[[MUL:.+]] = arith.muli %[[DIV]], %[[C8]]
// CHECK:         %[[REM:.+]] = arith.subi %[[FLAT]], %[[MUL]]
// CHECK:         return %[[DIV]], %[[REM]]
func.func @vector_linearize_offset_delinearize(%v0: vector<4xindex>, %v1: vector<4xindex>, %offsets: vector<4xindex>) -> (vector<4xindex>, vector<4xindex>) {
  %0 = affine.linearize_index [%v0, %v1] by (4, 8) : vector<4xindex>
  %1 = arith.addi %0, %offsets : vector<4xindex>
  %2:2 = affine.delinearize_index %1 into (4, 8) : vector<4xindex>, vector<4xindex>
  return %2#0, %2#1 : vector<4xindex>, vector<4xindex>
}
