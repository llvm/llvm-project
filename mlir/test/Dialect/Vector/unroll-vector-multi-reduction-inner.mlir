// RUN: mlir-opt --split-input-file %s -transform-preload-library='transform-library-paths=%p/td/unroll-multi-reduction.mlir' \
// RUN: -transform-interpreter=entry-point=unroll_multi_reduction_inner | FileCheck %s

//===----------------------------------------------------------------------===//
// Test UnrollVectorMultiReduction for Inner Reduction (Base Case)
//===----------------------------------------------------------------------===//

// The pattern recursively reduces rank until we reach 1D multi_reductions.
// For vector<2x3x5xf32> with reduction on dim 2:
// - First pass: unrolls along dim 0 (size 2), creating vector<3x5xf32> multi_reductions
// - Second pass: unrolls along dim 0 (size 3), creating vector<5xf32> multi_reductions
//
// The generated IR groups operations by phase:
// extracts (source) → extracts (acc) → multi_reductions → inserts

// CHECK-LABEL: func @unroll_vector_multi_reduction_inner(
// CHECK-SAME: %[[SOURCE:.+]]: vector<2x3x5xf32>,
// CHECK-SAME: %[[ACC:.+]]: vector<2x3xf32>
func.func @unroll_vector_multi_reduction_inner(%source: vector<2x3x5xf32>, %acc: vector<2x3xf32>) -> (vector<2x3xf32>) {
  // First slice [0, ...]: extracts → reductions → inserts
  // CHECK: vector.extract %[[SOURCE]][0, 0]
  // CHECK: vector.extract %[[SOURCE]][0, 1]
  // CHECK: vector.extract %[[SOURCE]][0, 2]
  // CHECK: vector.extract %[[ACC]][0, 0]
  // CHECK: vector.extract %[[ACC]][0, 1]
  // CHECK: vector.extract %[[ACC]][0, 2]
  // CHECK: vector.multi_reduction <add>, {{.*}} [0] : vector<5xf32> to f32
  // CHECK: vector.multi_reduction <add>, {{.*}} [0] : vector<5xf32> to f32
  // CHECK: vector.multi_reduction <add>, {{.*}} [0] : vector<5xf32> to f32
  // CHECK: vector.insert {{.*}} [0] : f32 into vector<3xf32>
  // CHECK: vector.insert {{.*}} [1] : f32 into vector<3xf32>
  // CHECK: vector.insert {{.*}} [2] : f32 into vector<3xf32>
  // Second slice [1, ...]: extracts → reductions → inserts
  // CHECK: vector.extract %[[SOURCE]][1, 0]
  // CHECK: vector.extract %[[SOURCE]][1, 1]
  // CHECK: vector.extract %[[SOURCE]][1, 2]
  // CHECK: vector.extract %[[ACC]][1, 0]
  // CHECK: vector.extract %[[ACC]][1, 1]
  // CHECK: vector.extract %[[ACC]][1, 2]
  // CHECK: vector.multi_reduction <add>, {{.*}} [0] : vector<5xf32> to f32
  // CHECK: vector.multi_reduction <add>, {{.*}} [0] : vector<5xf32> to f32
  // CHECK: vector.multi_reduction <add>, {{.*}} [0] : vector<5xf32> to f32
  // CHECK: vector.insert {{.*}} [0] : f32 into vector<3xf32>
  // CHECK: vector.insert {{.*}} [1] : f32 into vector<3xf32>
  // CHECK: vector.insert {{.*}} [2] : f32 into vector<3xf32>
  // Final inserts to assemble result
  // CHECK: vector.insert {{.*}} [0] : vector<3xf32> into vector<2x3xf32>
  // CHECK: vector.insert {{.*}} [1] : vector<3xf32> into vector<2x3xf32>
  // No original multi_reduction with [2] remains
  // CHECK-NOT: vector.multi_reduction <add>, {{.*}} [2]
  %1 = vector.multi_reduction <add>, %source, %acc [2] : vector<2x3x5xf32> to vector<2x3xf32>

  return %1 : vector<2x3xf32>
}

// -----

// CHECK-LABEL: func @unroll_vector_multi_reduction_inner_masked(
// CHECK-SAME: %[[SOURCE:.+]]: vector<2x3x5xf32>,
// CHECK-SAME: %[[MASK:.+]]: vector<2x3x5xi1>,
// CHECK-SAME: %[[ACC:.+]]: vector<2x3xf32>
func.func @unroll_vector_multi_reduction_inner_masked(%source: vector<2x3x5xf32>, %mask: vector<2x3x5xi1>, %acc: vector<2x3xf32>) -> (vector<2x3xf32>) {
  // First slice [0, ...]: extracts (source, acc, mask) → masked reductions → inserts
  // CHECK: vector.extract %[[SOURCE]][0, 0]
  // CHECK: vector.extract %[[SOURCE]][0, 1]
  // CHECK: vector.extract %[[SOURCE]][0, 2]
  // CHECK: vector.extract %[[ACC]][0, 0]
  // CHECK: vector.extract %[[ACC]][0, 1]
  // CHECK: vector.extract %[[ACC]][0, 2]
  // CHECK: vector.extract %[[MASK]][0, 0]
  // CHECK: vector.extract %[[MASK]][0, 1]
  // CHECK: vector.extract %[[MASK]][0, 2]
  // CHECK: vector.mask {{.*}} { vector.multi_reduction <add>, {{.*}} [0] : vector<5xf32> to f32 }
  // CHECK: vector.mask {{.*}} { vector.multi_reduction <add>, {{.*}} [0] : vector<5xf32> to f32 }
  // CHECK: vector.mask {{.*}} { vector.multi_reduction <add>, {{.*}} [0] : vector<5xf32> to f32 }
  // CHECK: vector.insert {{.*}} [0] : f32 into vector<3xf32>
  // CHECK: vector.insert {{.*}} [1] : f32 into vector<3xf32>
  // CHECK: vector.insert {{.*}} [2] : f32 into vector<3xf32>
  // Second slice [1, ...]: extracts → masked reductions → inserts
  // CHECK: vector.extract %[[SOURCE]][1, 0]
  // CHECK: vector.extract %[[SOURCE]][1, 1]
  // CHECK: vector.extract %[[SOURCE]][1, 2]
  // CHECK: vector.extract %[[ACC]][1, 0]
  // CHECK: vector.extract %[[ACC]][1, 1]
  // CHECK: vector.extract %[[ACC]][1, 2]
  // CHECK: vector.extract %[[MASK]][1, 0]
  // CHECK: vector.extract %[[MASK]][1, 1]
  // CHECK: vector.extract %[[MASK]][1, 2]
  // CHECK: vector.mask {{.*}} { vector.multi_reduction <add>, {{.*}} [0] : vector<5xf32> to f32 }
  // CHECK: vector.mask {{.*}} { vector.multi_reduction <add>, {{.*}} [0] : vector<5xf32> to f32 }
  // CHECK: vector.mask {{.*}} { vector.multi_reduction <add>, {{.*}} [0] : vector<5xf32> to f32 }
  // CHECK: vector.insert {{.*}} [0] : f32 into vector<3xf32>
  // CHECK: vector.insert {{.*}} [1] : f32 into vector<3xf32>
  // CHECK: vector.insert {{.*}} [2] : f32 into vector<3xf32>
  // Final inserts to assemble result
  // CHECK: vector.insert {{.*}} [0] : vector<3xf32> into vector<2x3xf32>
  // CHECK: vector.insert {{.*}} [1] : vector<3xf32> into vector<2x3xf32>
  // No original multi_reduction with [2] remains
  // CHECK-NOT: vector.multi_reduction <add>, {{.*}} [2]

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
// - Unrolls along dim 0 (size 2), creating vector<3x5xf32> multi_reductions
// - Each new multi_reduction has reduction dims [0, 1] (shifted from [1, 2])

// CHECK-LABEL: func @unroll_vector_multi_reduction_inner_general(
// CHECK-SAME: %[[SOURCE:.+]]: vector<2x3x5xf32>,
// CHECK-SAME: %[[ACC:.+]]: vector<2xf32>
func.func @unroll_vector_multi_reduction_inner_general(%source: vector<2x3x5xf32>, %acc: vector<2xf32>) -> (vector<2xf32>) {
  // CHECK-DAG: %[[VEC_0:.+]] = vector.extract %[[SOURCE]][0] : vector<3x5xf32> from vector<2x3x5xf32>
  // CHECK-DAG: %[[VEC_1:.+]] = vector.extract %[[SOURCE]][1] : vector<3x5xf32> from vector<2x3x5xf32>

  // CHECK-DAG: %[[ACC_0:.+]] = vector.extract %[[ACC]][0] : f32 from vector<2xf32>
  // CHECK-DAG: %[[ACC_1:.+]] = vector.extract %[[ACC]][1] : f32 from vector<2xf32>

  // CHECK: %[[RED_0:.+]] = vector.multi_reduction <add>, %[[VEC_0]], %[[ACC_0]] [0, 1] : vector<3x5xf32> to f32
  // CHECK: %[[RED_1:.+]] = vector.multi_reduction <add>, %[[VEC_1]], %[[ACC_1]] [0, 1] : vector<3x5xf32> to f32
  // CHECK: %[[INSERT_0:.+]] = vector.insert %[[RED_0]], %{{.*}} [0] : f32 into vector<2xf32>
  // CHECK: %[[INSERT_1:.+]] = vector.insert %[[RED_1]], %[[INSERT_0]] [1] : f32 into vector<2xf32>
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
  // CHECK-DAG: %[[VEC_0:.+]] = vector.extract %[[SOURCE]][0] : vector<3x5xf32> from vector<2x3x5xf32>
  // CHECK-DAG: %[[VEC_1:.+]] = vector.extract %[[SOURCE]][1] : vector<3x5xf32> from vector<2x3x5xf32>

  // CHECK-DAG: %[[ACC_0:.+]] = vector.extract %[[ACC]][0] : f32 from vector<2xf32>
  // CHECK-DAG: %[[ACC_1:.+]] = vector.extract %[[ACC]][1] : f32 from vector<2xf32>

  // CHECK-DAG: %[[MASK_0:.+]] = vector.extract %[[MASK]][0] : vector<3x5xi1> from vector<2x3x5xi1>
  // CHECK-DAG: %[[MASK_1:.+]] = vector.extract %[[MASK]][1] : vector<3x5xi1> from vector<2x3x5xi1>

  // CHECK: %[[RED_0:.+]] = vector.mask %[[MASK_0]] { vector.multi_reduction <add>, %[[VEC_0]], %[[ACC_0]] [0, 1] : vector<3x5xf32> to f32 } : vector<3x5xi1> -> f32
  // CHECK: %[[RED_1:.+]] = vector.mask %[[MASK_1]] { vector.multi_reduction <add>, %[[VEC_1]], %[[ACC_1]] [0, 1] : vector<3x5xf32> to f32 } : vector<3x5xi1> -> f32
  // CHECK: %[[INSERT_0:.+]] = vector.insert %[[RED_0]], %{{.*}} [0] : f32 into vector<2xf32>
  // CHECK: %[[INSERT_1:.+]] = vector.insert %[[RED_1]], %[[INSERT_0]] [1] : f32 into vector<2xf32>

  %0 = vector.mask %mask {
    %1 = vector.multi_reduction <add>, %source, %acc [1, 2] : vector<2x3x5xf32> to vector<2xf32>
  } : vector<2x3x5xi1> -> vector<2xf32>

  // CHECK: return %[[INSERT_1]]
  return %0 : vector<2xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Negative Test: Rank-1 multi_reduction should NOT be matched by UnrollMultiReductionInner
//===----------------------------------------------------------------------===//

// UnrollMultiReductionInner requires srcRank >= 2, so rank-1 should not match.
// The op should remain unchanged.

// CHECK-LABEL: func @unroll_vector_multi_reduction_inner_rank1_negative(
// CHECK-SAME: %[[SOURCE:.+]]: vector<8xf32>,
// CHECK-SAME: %[[ACC:.+]]: f32
func.func @unroll_vector_multi_reduction_inner_rank1_negative(%source: vector<8xf32>, %acc: f32) -> f32 {
  // CHECK: %[[RESULT:.+]] = vector.multi_reduction <add>, %[[SOURCE]], %[[ACC]] [0] : vector<8xf32> to f32
  %0 = vector.multi_reduction <add>, %source, %acc [0] : vector<8xf32> to f32
  // CHECK: return %[[RESULT]]
  return %0 : f32
}
