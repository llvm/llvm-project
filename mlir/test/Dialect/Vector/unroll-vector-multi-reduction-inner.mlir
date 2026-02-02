// RUN: mlir-opt --split-input-file %s -transform-preload-library='transform-library-paths=%p/td/unroll-multi-reduction.mlir' \
// RUN: -transform-interpreter=entry-point=unroll_multi_reduction_inner | FileCheck %s

//===----------------------------------------------------------------------===//
// Test UnrollVectorMultiReduction for Inner Reduction (Base Case)
//===----------------------------------------------------------------------===//

// The pattern recursively reduces rank until we reach 1D, then converts to
// vector.reduction via OneDimMultiReductionToReduction.
// For vector<2x3x5xf32> with reduction on dim 2:
// - First pass: unrolls along dim 0 (size 2), creating vector<3x5xf32> multi_reductions
// - Second pass: unrolls along dim 0 (size 3), creating vector<5xf32> multi_reductions
// - Final: 1-D multi_reductions are converted to vector.reduction
//
// The generated IR groups operations by phase:
// extracts (source) → extracts (acc) → reductions → inserts

// CHECK-LABEL: func @unroll_vector_multi_reduction_inner(
// CHECK-SAME: %[[SOURCE:.+]]: vector<2x3x5xf32>,
// CHECK-SAME: %[[ACC:.+]]: vector<2x3xf32>
func.func @unroll_vector_multi_reduction_inner(%source: vector<2x3x5xf32>, %acc: vector<2x3xf32>) -> (vector<2x3xf32>) {
  // CHECK: vector.extract %[[SOURCE]][0, 0]
  // CHECK: vector.extract %[[SOURCE]][0, 1]
  // CHECK: vector.extract %[[SOURCE]][0, 2]
  // CHECK: vector.extract %[[ACC]][0, 0]
  // CHECK: vector.extract %[[ACC]][0, 1]
  // CHECK: vector.extract %[[ACC]][0, 2]
  // CHECK: vector.reduction <add>, {{.*}} : vector<5xf32> into f32
  // CHECK: vector.reduction <add>, {{.*}} : vector<5xf32> into f32
  // CHECK: vector.reduction <add>, {{.*}} : vector<5xf32> into f32
  // CHECK: vector.insert {{.*}} [0] : f32 into vector<3xf32>
  // CHECK: vector.insert {{.*}} [1] : f32 into vector<3xf32>
  // CHECK: vector.insert {{.*}} [2] : f32 into vector<3xf32>
  // CHECK: vector.extract %[[SOURCE]][1, 0]
  // CHECK: vector.extract %[[SOURCE]][1, 1]
  // CHECK: vector.extract %[[SOURCE]][1, 2]
  // CHECK: vector.extract %[[ACC]][1, 0]
  // CHECK: vector.extract %[[ACC]][1, 1]
  // CHECK: vector.extract %[[ACC]][1, 2]
  // CHECK: vector.reduction <add>, {{.*}} : vector<5xf32> into f32
  // CHECK: vector.reduction <add>, {{.*}} : vector<5xf32> into f32
  // CHECK: vector.reduction <add>, {{.*}} : vector<5xf32> into f32
  // CHECK: vector.insert {{.*}} [0] : f32 into vector<3xf32>
  // CHECK: vector.insert {{.*}} [1] : f32 into vector<3xf32>
  // CHECK: vector.insert {{.*}} [2] : f32 into vector<3xf32>
  // CHECK: vector.insert {{.*}} [0] : vector<3xf32> into vector<2x3xf32>
  // CHECK: vector.insert {{.*}} [1] : vector<3xf32> into vector<2x3xf32>
  // CHECK-NOT: vector.multi_reduction
  %1 = vector.multi_reduction <add>, %source, %acc [2] : vector<2x3x5xf32> to vector<2x3xf32>

  return %1 : vector<2x3xf32>
}

// -----

// CHECK-LABEL: func @unroll_vector_multi_reduction_inner_masked(
// CHECK-SAME: %[[SOURCE:.+]]: vector<2x3x5xf32>,
// CHECK-SAME: %[[MASK:.+]]: vector<2x3x5xi1>,
// CHECK-SAME: %[[ACC:.+]]: vector<2x3xf32>
func.func @unroll_vector_multi_reduction_inner_masked(%source: vector<2x3x5xf32>, %mask: vector<2x3x5xi1>, %acc: vector<2x3xf32>) -> (vector<2x3xf32>) {
  // CHECK: vector.extract %[[SOURCE]][0, 0]
  // CHECK: vector.extract %[[SOURCE]][0, 1]
  // CHECK: vector.extract %[[SOURCE]][0, 2]
  // CHECK: vector.extract %[[ACC]][0, 0]
  // CHECK: vector.extract %[[ACC]][0, 1]
  // CHECK: vector.extract %[[ACC]][0, 2]
  // CHECK: vector.extract %[[MASK]][0, 0]
  // CHECK: vector.extract %[[MASK]][0, 1]
  // CHECK: vector.extract %[[MASK]][0, 2]
  // CHECK: vector.mask {{.*}} { vector.reduction <add>, {{.*}} : vector<5xf32> into f32 }
  // CHECK: vector.mask {{.*}} { vector.reduction <add>, {{.*}} : vector<5xf32> into f32 }
  // CHECK: vector.mask {{.*}} { vector.reduction <add>, {{.*}} : vector<5xf32> into f32 }
  // CHECK: vector.insert {{.*}} [0] : f32 into vector<3xf32>
  // CHECK: vector.insert {{.*}} [1] : f32 into vector<3xf32>
  // CHECK: vector.insert {{.*}} [2] : f32 into vector<3xf32>
  // CHECK: vector.extract %[[SOURCE]][1, 0]
  // CHECK: vector.extract %[[SOURCE]][1, 1]
  // CHECK: vector.extract %[[SOURCE]][1, 2]
  // CHECK: vector.extract %[[ACC]][1, 0]
  // CHECK: vector.extract %[[ACC]][1, 1]
  // CHECK: vector.extract %[[ACC]][1, 2]
  // CHECK: vector.extract %[[MASK]][1, 0]
  // CHECK: vector.extract %[[MASK]][1, 1]
  // CHECK: vector.extract %[[MASK]][1, 2]
  // CHECK: vector.mask {{.*}} { vector.reduction <add>, {{.*}} : vector<5xf32> into f32 }
  // CHECK: vector.mask {{.*}} { vector.reduction <add>, {{.*}} : vector<5xf32> into f32 }
  // CHECK: vector.mask {{.*}} { vector.reduction <add>, {{.*}} : vector<5xf32> into f32 }
  // CHECK: vector.insert {{.*}} [0] : f32 into vector<3xf32>
  // CHECK: vector.insert {{.*}} [1] : f32 into vector<3xf32>
  // CHECK: vector.insert {{.*}} [2] : f32 into vector<3xf32>
  // CHECK: vector.insert {{.*}} [0] : vector<3xf32> into vector<2x3xf32>
  // CHECK: vector.insert {{.*}} [1] : vector<3xf32> into vector<2x3xf32>
  // CHECK-NOT: vector.multi_reduction

  %0 = vector.mask %mask {
    %1 = vector.multi_reduction <add>, %source, %acc [2] : vector<2x3x5xf32> to vector<2x3xf32>
  } : vector<2x3x5xi1> -> vector<2x3xf32>

  return %0 : vector<2x3xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Test UnrollVectorMultiReduction for Inner Reduction (General Case)
//===----------------------------------------------------------------------===//

// The general case handles multiple reduction dimensions.
// For vector<2x3x5xf32> with reduction on dims [1, 2]:
// - First: UnrollInnerReductionAlongOuterParallel unrolls along dim 0 (size 2),
//   creating vector<3x5xf32> multi_reductions with dims [0, 1]
// - Then: UnrollMultiReductionOuterGeneralCase handles these (outermost is now
//   reduction), extracting along dim 0 and chaining 1-D multi_reductions
// - Finally: OneDimMultiReductionToReduction converts to vector.reduction

// CHECK-LABEL: func @unroll_vector_multi_reduction_inner_general(
// CHECK-SAME: %[[SOURCE:.+]]: vector<2x3x5xf32>,
// CHECK-SAME: %[[ACC:.+]]: vector<2xf32>
func.func @unroll_vector_multi_reduction_inner_general(%source: vector<2x3x5xf32>, %acc: vector<2xf32>) -> (vector<2xf32>) {
  // CHECK-DAG: %[[ACC_0:.+]] = vector.extract %[[ACC]][0] : f32 from vector<2xf32>
  // CHECK-DAG: %[[ACC_1:.+]] = vector.extract %[[ACC]][1] : f32 from vector<2xf32>
  // CHECK: vector.extract %[[SOURCE]][0, 0] : vector<5xf32>
  // CHECK: vector.extract %[[SOURCE]][0, 1] : vector<5xf32>
  // CHECK: vector.extract %[[SOURCE]][0, 2] : vector<5xf32>
  // CHECK: %[[R0_0:.+]] = vector.reduction <add>, {{.*}}, %[[ACC_0]] : vector<5xf32> into f32
  // CHECK: %[[R0_1:.+]] = vector.reduction <add>, {{.*}}, %[[R0_0]] : vector<5xf32> into f32
  // CHECK: %[[R0_2:.+]] = vector.reduction <add>, {{.*}}, %[[R0_1]] : vector<5xf32> into f32
  // CHECK: vector.extract %[[SOURCE]][1, 0] : vector<5xf32>
  // CHECK: vector.extract %[[SOURCE]][1, 1] : vector<5xf32>
  // CHECK: vector.extract %[[SOURCE]][1, 2] : vector<5xf32>
  // CHECK: %[[R1_0:.+]] = vector.reduction <add>, {{.*}}, %[[ACC_1]] : vector<5xf32> into f32
  // CHECK: %[[R1_1:.+]] = vector.reduction <add>, {{.*}}, %[[R1_0]] : vector<5xf32> into f32
  // CHECK: %[[R1_2:.+]] = vector.reduction <add>, {{.*}}, %[[R1_1]] : vector<5xf32> into f32
  // CHECK: %[[INSERT_0:.+]] = vector.insert %[[R0_2]], %{{.*}} [0] : f32 into vector<2xf32>
  // CHECK: %[[INSERT_1:.+]] = vector.insert %[[R1_2]], %[[INSERT_0]] [1] : f32 into vector<2xf32>
  // CHECK-NOT: vector.multi_reduction
  %1 = vector.multi_reduction <add>, %source, %acc [1, 2] : vector<2x3x5xf32> to vector<2xf32>

  // CHECK: return %[[INSERT_1]]
  return %1 : vector<2xf32>
}

// -----

// CHECK-LABEL: func @unroll_vector_multi_reduction_inner_general_masked(
// CHECK-SAME: %[[SOURCE:.+]]: vector<2x3x5xf32>,
// CHECK-SAME: %[[MASK:.+]]: vector<2x3x5xi1>,
// CHECK-SAME: %[[ACC:.+]]: vector<2xf32>
func.func @unroll_vector_multi_reduction_inner_general_masked(%source: vector<2x3x5xf32>, %mask: vector<2x3x5xi1>, %acc: vector<2xf32>) -> (vector<2xf32>) {
  // CHECK-DAG: %[[ACC_0:.+]] = vector.extract %[[ACC]][0] : f32 from vector<2xf32>
  // CHECK-DAG: %[[ACC_1:.+]] = vector.extract %[[ACC]][1] : f32 from vector<2xf32>
  // CHECK: vector.extract %[[SOURCE]][0, 0] : vector<5xf32>
  // CHECK: vector.extract %[[SOURCE]][0, 1] : vector<5xf32>
  // CHECK: vector.extract %[[SOURCE]][0, 2] : vector<5xf32>
  // CHECK: vector.extract %[[MASK]][0, 0] : vector<5xi1>
  // CHECK: vector.extract %[[MASK]][0, 1] : vector<5xi1>
  // CHECK: vector.extract %[[MASK]][0, 2] : vector<5xi1>
  // CHECK: %[[R0_0:.+]] = vector.mask {{.*}} { vector.reduction <add>, {{.*}}, %[[ACC_0]] : vector<5xf32> into f32 }
  // CHECK: %[[R0_1:.+]] = vector.mask {{.*}} { vector.reduction <add>, {{.*}}, %[[R0_0]] : vector<5xf32> into f32 }
  // CHECK: %[[R0_2:.+]] = vector.mask {{.*}} { vector.reduction <add>, {{.*}}, %[[R0_1]] : vector<5xf32> into f32 }
  // CHECK: vector.extract %[[SOURCE]][1, 0] : vector<5xf32>
  // CHECK: vector.extract %[[SOURCE]][1, 1] : vector<5xf32>
  // CHECK: vector.extract %[[SOURCE]][1, 2] : vector<5xf32>
  // CHECK: vector.extract %[[MASK]][1, 0] : vector<5xi1>
  // CHECK: vector.extract %[[MASK]][1, 1] : vector<5xi1>
  // CHECK: vector.extract %[[MASK]][1, 2] : vector<5xi1>
  // CHECK: %[[R1_0:.+]] = vector.mask {{.*}} { vector.reduction <add>, {{.*}}, %[[ACC_1]] : vector<5xf32> into f32 }
  // CHECK: %[[R1_1:.+]] = vector.mask {{.*}} { vector.reduction <add>, {{.*}}, %[[R1_0]] : vector<5xf32> into f32 }
  // CHECK: %[[R1_2:.+]] = vector.mask {{.*}} { vector.reduction <add>, {{.*}}, %[[R1_1]] : vector<5xf32> into f32 }
  // CHECK: %[[INSERT_0:.+]] = vector.insert %[[R0_2]], %{{.*}} [0] : f32 into vector<2xf32>
  // CHECK: %[[INSERT_1:.+]] = vector.insert %[[R1_2]], %[[INSERT_0]] [1] : f32 into vector<2xf32>
  // CHECK-NOT: vector.multi_reduction

  %0 = vector.mask %mask {
    %1 = vector.multi_reduction <add>, %source, %acc [1, 2] : vector<2x3x5xf32> to vector<2xf32>
  } : vector<2x3x5xi1> -> vector<2xf32>

  // CHECK: return %[[INSERT_1]]
  return %0 : vector<2xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Test 1-D multi_reduction to vector.reduction conversion
//===----------------------------------------------------------------------===//

// OneDimMultiReductionToReduction converts rank-1 multi_reduction directly
// to vector.reduction, which preserves the reduction semantic for backends.

// CHECK-LABEL: func @unroll_vector_multi_reduction_inner_1d(
// CHECK-SAME: %[[SOURCE:.+]]: vector<8xf32>,
// CHECK-SAME: %[[ACC:.+]]: f32
func.func @unroll_vector_multi_reduction_inner_1d(%source: vector<8xf32>, %acc: f32) -> f32 {
  // CHECK: %[[RESULT:.+]] = vector.reduction <add>, %[[SOURCE]], %[[ACC]] : vector<8xf32> into f32
  %0 = vector.multi_reduction <add>, %source, %acc [0] : vector<8xf32> to f32
  // CHECK: return %[[RESULT]]
  return %0 : f32
}

// -----

// CHECK-LABEL: func @unroll_vector_multi_reduction_inner_1d_masked(
// CHECK-SAME: %[[SOURCE:.+]]: vector<8xf32>,
// CHECK-SAME: %[[MASK:.+]]: vector<8xi1>,
// CHECK-SAME: %[[ACC:.+]]: f32
func.func @unroll_vector_multi_reduction_inner_1d_masked(%source: vector<8xf32>, %mask: vector<8xi1>, %acc: f32) -> f32 {
  // CHECK: %[[RESULT:.+]] = vector.mask %[[MASK]] { vector.reduction <add>, %[[SOURCE]], %[[ACC]] : vector<8xf32> into f32 } : vector<8xi1> -> f32
  %0 = vector.mask %mask {
    vector.multi_reduction <add>, %source, %acc [0] : vector<8xf32> to f32
  } : vector<8xi1> -> f32
  // CHECK: return %[[RESULT]]
  return %0 : f32
}
