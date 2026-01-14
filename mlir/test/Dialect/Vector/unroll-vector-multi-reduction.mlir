// RUN: mlir-opt --split-input-file %s -transform-preload-library='transform-library-paths=%p/td/unroll-multi-reduction.mlir' \
// RUN: -transform-interpreter=entry-point=unroll_multi_reduction | FileCheck %s

//===----------------------------------------------------------------------===//
// Test UnrollVectorMultiReduction
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @unroll_vector_multi_reduction(
// CHECK-SAME: %[[SOURCE:.+]]: vector<2x3x5xf32>,
// CHECK-SAME: %[[ACC:.+]]: vector<3x5xf32>
func.func @unroll_vector_multi_reduction(%source: vector<2x3x5xf32>, %acc: vector<3x5xf32>) -> (vector<3x5xf32>) {
  // CHECK-DAG: %[[VEC_0:.+]] = vector.extract %[[SOURCE]][0] : vector<3x5xf32> from vector<2x3x5xf32>
  // CHECK-DAG: %[[VEC_1:.+]] = vector.extract %[[SOURCE]][1] : vector<3x5xf32> from vector<2x3x5xf32>

  // CHECK: %[[RES_0:.+]] = arith.addf %[[VEC_0]], %[[ACC]] : vector<3x5xf32>
  // CHECK: %[[RES_1:.+]] = arith.addf %[[VEC_1]], %[[RES_0]] : vector<3x5xf32>
  %1 = vector.multi_reduction <add>, %source, %acc [0] : vector<2x3x5xf32> to vector<3x5xf32>

  // CHECK: return %[[RES_1]]
  return %1 : vector<3x5xf32>
}

// -----

// CHECK-LABEL: func @unroll_vector_multi_reduction_masked(
// CHECK-SAME: %[[SOURCE:.+]]: vector<2x3x5xf32>,
// CHECK-SAME: %[[MASK:.+]]: vector<2x3x5xi1>,
// CHECK-SAME: %[[ACC:.+]]: vector<3x5xf32>
func.func @unroll_vector_multi_reduction_masked(%source: vector<2x3x5xf32>, %mask: vector<2x3x5xi1>, %acc: vector<3x5xf32>) -> (vector<3x5xf32>) {
  // CHECK-DAG: %[[VEC_0:.+]] = vector.extract %[[SOURCE]][0] : vector<3x5xf32> from vector<2x3x5xf32>
  // CHECK-DAG: %[[VEC_1:.+]] = vector.extract %[[SOURCE]][1] : vector<3x5xf32> from vector<2x3x5xf32>

  // CHECK-DAG: %[[MASK_0:.+]] = vector.extract %[[MASK]][0] : vector<3x5xi1> from vector<2x3x5xi1>
  // CHECK-DAG: %[[MASK_1:.+]] = vector.extract %[[MASK]][1] : vector<3x5xi1> from vector<2x3x5xi1>

  // CHECK: %[[RES_0:.+]] = arith.addf %[[VEC_0]], %[[ACC]] : vector<3x5xf32>
  // CHECK: %[[RES_MASKED_0:.+]] = arith.select %[[MASK_0]], %[[RES_0]], %[[ACC]] : vector<3x5xi1>, vector<3x5xf32>

  // CHECK: %[[RES_1:.+]] = arith.addf %[[VEC_1]], %[[RES_MASKED_0]] : vector<3x5xf32>
  // CHECK: %[[RES_MASKED_1:.+]] = arith.select %[[MASK_1]], %[[RES_1]], %[[RES_MASKED_0]] : vector<3x5xi1>, vector<3x5xf32>

  %0 = vector.mask %mask {
    %1 = vector.multi_reduction <add>, %source, %acc [0] : vector<2x3x5xf32> to vector<3x5xf32>
  } : vector<2x3x5xi1> -> vector<3x5xf32>

  // CHECK: return %[[RES_MASKED_1]]
  return %0 : vector<3x5xf32>
}

// -----

// CHECK-LABEL: func @unroll_vector_multi_reduction_general(
// CHECK-SAME: %[[SOURCE:.+]]: vector<2x3x5xf32>,
// CHECK-SAME: %[[ACC:.+]]: vector<3xf32>
func.func @unroll_vector_multi_reduction_general(%source: vector<2x3x5xf32>, %acc: vector<3xf32>) -> (vector<3xf32>) {

  // CHECK-DAG: %[[VEC_0:.+]] = vector.extract %[[SOURCE]][0] : vector<3x5xf32> from vector<2x3x5xf32>
  // CHECK-DAG: %[[VEC_1:.+]] = vector.extract %[[SOURCE]][1] : vector<3x5xf32> from vector<2x3x5xf32>

  // CHECK: %[[RES_0:.+]] = vector.multi_reduction <add>, %[[VEC_0]], %[[ACC]] [1] : vector<3x5xf32> to vector<3xf32>
  // CHECK: %[[RES_1:.+]] = vector.multi_reduction <add>, %[[VEC_1]], %[[RES_0]] [1] : vector<3x5xf32> to vector<3xf32>

  %1 = vector.multi_reduction <add>, %source, %acc [0, 2] : vector<2x3x5xf32> to vector<3xf32>

  // CHECK: return %[[RES_1]]
  return %1 : vector<3xf32>
}

// -----

// CHECK-LABEL: func @unroll_vector_multi_reduction_general_masked(
// CHECK-SAME: %[[SOURCE:.+]]: vector<2x3x5xf32>,
// CHECK-SAME: %[[MASK:.+]]: vector<2x3x5xi1>,
// CHECK-SAME: %[[ACC:.+]]: vector<3xf32>
func.func @unroll_vector_multi_reduction_general_masked(%source: vector<2x3x5xf32>, %mask: vector<2x3x5xi1>, %acc: vector<3xf32>) -> (vector<3xf32>) {

  // CHECK-DAG: %[[VEC_0:.+]] = vector.extract %[[SOURCE]][0] : vector<3x5xf32> from vector<2x3x5xf32>
  // CHECK-DAG: %[[VEC_1:.+]] = vector.extract %[[SOURCE]][1] : vector<3x5xf32> from vector<2x3x5xf32>

  // CHECK-DAG: %[[MASK_0:.+]] = vector.extract %[[MASK]][0] : vector<3x5xi1> from vector<2x3x5xi1>
  // CHECK-DAG: %[[MASK_1:.+]] = vector.extract %[[MASK]][1] : vector<3x5xi1> from vector<2x3x5xi1>

  // CHECK: %[[RES_0:.+]] = vector.mask %[[MASK_0]] { vector.multi_reduction <add>, %[[VEC_0]], %[[ACC]] [1] : vector<3x5xf32> to vector<3xf32> } : vector<3x5xi1> -> vector<3xf32>
  // CHECK: %[[RES_1:.+]] = vector.mask %[[MASK_1]] { vector.multi_reduction <add>, %[[VEC_1]], %[[RES_0]] [1] : vector<3x5xf32> to vector<3xf32> } : vector<3x5xi1> -> vector<3xf32>

  %0 = vector.mask %mask {
    %1 = vector.multi_reduction <add>, %source, %acc [0, 2] : vector<2x3x5xf32> to vector<3xf32>
  } : vector<2x3x5xi1> -> vector<3xf32>

  // CHECK: return %[[RES_1]]
  return %0 : vector<3xf32>
}
