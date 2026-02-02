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

// Multiple reduction dims with outermost as reduction. Fully lowers to
// vector.reduction by combining outer and inner unrolling patterns.

// CHECK-LABEL: func @unroll_vector_multi_reduction_general(
// CHECK-SAME: %[[SOURCE:.+]]: vector<2x3x5xf32>,
// CHECK-SAME: %[[ACC:.+]]: vector<3xf32>
func.func @unroll_vector_multi_reduction_general(%source: vector<2x3x5xf32>, %acc: vector<3xf32>) -> (vector<3xf32>) {
  // CHECK: vector.extract %[[SOURCE]][0, 0] : vector<5xf32>
  // CHECK: vector.extract %[[SOURCE]][0, 1] : vector<5xf32>
  // CHECK: vector.extract %[[SOURCE]][0, 2] : vector<5xf32>
  // CHECK-DAG: %[[ACC_0:.+]] = vector.extract %[[ACC]][0] : f32
  // CHECK-DAG: %[[ACC_1:.+]] = vector.extract %[[ACC]][1] : f32
  // CHECK-DAG: %[[ACC_2:.+]] = vector.extract %[[ACC]][2] : f32
  // CHECK: %[[R0_0:.+]] = vector.reduction <add>, {{.*}}, %[[ACC_0]] : vector<5xf32> into f32
  // CHECK: %[[R0_1:.+]] = vector.reduction <add>, {{.*}}, %[[ACC_1]] : vector<5xf32> into f32
  // CHECK: %[[R0_2:.+]] = vector.reduction <add>, {{.*}}, %[[ACC_2]] : vector<5xf32> into f32
  // CHECK: vector.extract %[[SOURCE]][1, 0] : vector<5xf32>
  // CHECK: vector.extract %[[SOURCE]][1, 1] : vector<5xf32>
  // CHECK: vector.extract %[[SOURCE]][1, 2] : vector<5xf32>
  // CHECK: %[[R1_0:.+]] = vector.reduction <add>, {{.*}}, %[[R0_0]] : vector<5xf32> into f32
  // CHECK: %[[R1_1:.+]] = vector.reduction <add>, {{.*}}, %[[R0_1]] : vector<5xf32> into f32
  // CHECK: %[[R1_2:.+]] = vector.reduction <add>, {{.*}}, %[[R0_2]] : vector<5xf32> into f32
  // CHECK: %[[INSERT_0:.+]] = vector.insert %[[R1_0]], %{{.*}} [0] : f32 into vector<3xf32>
  // CHECK: %[[INSERT_1:.+]] = vector.insert %[[R1_1]], %[[INSERT_0]] [1] : f32 into vector<3xf32>
  // CHECK: %[[INSERT_2:.+]] = vector.insert %[[R1_2]], %[[INSERT_1]] [2] : f32 into vector<3xf32>
  // CHECK-NOT: vector.multi_reduction
  %1 = vector.multi_reduction <add>, %source, %acc [0, 2] : vector<2x3x5xf32> to vector<3xf32>

  // CHECK: return %[[INSERT_2]]
  return %1 : vector<3xf32>
}

// -----

// CHECK-LABEL: func @unroll_vector_multi_reduction_general_masked(
// CHECK-SAME: %[[SOURCE:.+]]: vector<2x3x5xf32>,
// CHECK-SAME: %[[MASK:.+]]: vector<2x3x5xi1>,
// CHECK-SAME: %[[ACC:.+]]: vector<3xf32>
func.func @unroll_vector_multi_reduction_general_masked(%source: vector<2x3x5xf32>, %mask: vector<2x3x5xi1>, %acc: vector<3xf32>) -> (vector<3xf32>) {
  // CHECK: vector.extract %[[SOURCE]][0, 0] : vector<5xf32>
  // CHECK: vector.extract %[[SOURCE]][0, 1] : vector<5xf32>
  // CHECK: vector.extract %[[SOURCE]][0, 2] : vector<5xf32>
  // CHECK-DAG: %[[ACC_0:.+]] = vector.extract %[[ACC]][0] : f32
  // CHECK-DAG: %[[ACC_1:.+]] = vector.extract %[[ACC]][1] : f32
  // CHECK-DAG: %[[ACC_2:.+]] = vector.extract %[[ACC]][2] : f32
  // CHECK: vector.extract %[[MASK]][0, 0] : vector<5xi1>
  // CHECK: vector.extract %[[MASK]][0, 1] : vector<5xi1>
  // CHECK: vector.extract %[[MASK]][0, 2] : vector<5xi1>
  // CHECK: %[[R0_0:.+]] = vector.mask {{.*}} { vector.reduction <add>, {{.*}}, %[[ACC_0]] : vector<5xf32> into f32 }
  // CHECK: %[[R0_1:.+]] = vector.mask {{.*}} { vector.reduction <add>, {{.*}}, %[[ACC_1]] : vector<5xf32> into f32 }
  // CHECK: %[[R0_2:.+]] = vector.mask {{.*}} { vector.reduction <add>, {{.*}}, %[[ACC_2]] : vector<5xf32> into f32 }
  // CHECK: vector.extract %[[SOURCE]][1, 0] : vector<5xf32>
  // CHECK: vector.extract %[[SOURCE]][1, 1] : vector<5xf32>
  // CHECK: vector.extract %[[SOURCE]][1, 2] : vector<5xf32>
  // CHECK: vector.extract %[[MASK]][1, 0] : vector<5xi1>
  // CHECK: vector.extract %[[MASK]][1, 1] : vector<5xi1>
  // CHECK: vector.extract %[[MASK]][1, 2] : vector<5xi1>
  // CHECK: %[[R1_0:.+]] = vector.mask {{.*}} { vector.reduction <add>, {{.*}}, %[[R0_0]] : vector<5xf32> into f32 }
  // CHECK: %[[R1_1:.+]] = vector.mask {{.*}} { vector.reduction <add>, {{.*}}, %[[R0_1]] : vector<5xf32> into f32 }
  // CHECK: %[[R1_2:.+]] = vector.mask {{.*}} { vector.reduction <add>, {{.*}}, %[[R0_2]] : vector<5xf32> into f32 }
  // CHECK: %[[INSERT_0:.+]] = vector.insert %[[R1_0]], %{{.*}} [0] : f32 into vector<3xf32>
  // CHECK: %[[INSERT_1:.+]] = vector.insert %[[R1_1]], %[[INSERT_0]] [1] : f32 into vector<3xf32>
  // CHECK: %[[INSERT_2:.+]] = vector.insert %[[R1_2]], %[[INSERT_1]] [2] : f32 into vector<3xf32>
  // CHECK-NOT: vector.multi_reduction

  %0 = vector.mask %mask {
    %1 = vector.multi_reduction <add>, %source, %acc [0, 2] : vector<2x3x5xf32> to vector<3xf32>
  } : vector<2x3x5xi1> -> vector<3xf32>

  // CHECK: return %[[INSERT_2]]
  return %0 : vector<3xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Test 1-D multi_reduction to vector.reduction conversion
//===----------------------------------------------------------------------===//

// OneDimMultiReductionToReduction converts rank-1 multi_reduction directly
// to vector.reduction, which preserves the reduction semantic for backends.

// CHECK-LABEL: func @unroll_vector_multi_reduction_1d(
// CHECK-SAME: %[[SOURCE:.+]]: vector<8xf32>,
// CHECK-SAME: %[[ACC:.+]]: f32
func.func @unroll_vector_multi_reduction_1d(%source: vector<8xf32>, %acc: f32) -> f32 {
  // CHECK: %[[RESULT:.+]] = vector.reduction <add>, %[[SOURCE]], %[[ACC]] : vector<8xf32> into f32
  %0 = vector.multi_reduction <add>, %source, %acc [0] : vector<8xf32> to f32
  // CHECK: return %[[RESULT]]
  return %0 : f32
}

// -----

// CHECK-LABEL: func @unroll_vector_multi_reduction_1d_masked(
// CHECK-SAME: %[[SOURCE:.+]]: vector<8xf32>,
// CHECK-SAME: %[[MASK:.+]]: vector<8xi1>,
// CHECK-SAME: %[[ACC:.+]]: f32
func.func @unroll_vector_multi_reduction_1d_masked(%source: vector<8xf32>, %mask: vector<8xi1>, %acc: f32) -> f32 {
  // CHECK: %[[RESULT:.+]] = vector.mask %[[MASK]] { vector.reduction <add>, %[[SOURCE]], %[[ACC]] : vector<8xf32> into f32 } : vector<8xi1> -> f32
  %0 = vector.mask %mask {
    vector.multi_reduction <add>, %source, %acc [0] : vector<8xf32> to f32
  } : vector<8xi1> -> f32
  // CHECK: return %[[RESULT]]
  return %0 : f32
}
