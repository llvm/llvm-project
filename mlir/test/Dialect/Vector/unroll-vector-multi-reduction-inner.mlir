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
