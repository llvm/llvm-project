// RUN: mlir-opt %s --transform-interpreter --split-input-file | FileCheck %s

/// Tests for `vector.contract` -> `vector.outerproduct` transformations for
/// matmul operations:
///   C += A * B.
/// (A, B and C are 2-d matrices). ATM three different variants / are tested:
///   * plain (no mask, fixed-wdith vectors),
///   * masked (fixed-width vectors,
///   * scalable (mask + scalable vectors).
/// In order for the "vector.contract -> vector.outerproduct" patterns to work,
/// only the non-reduction dimension can be scalable (*). For matmul operations
/// that is set to be the N dimension (i.e. rows of the output matrix), which
/// matches how matrix multiplication are normally implemented for e.g.
/// Arm SVE. However, making the M dimension scalable (i.e. columns of the
/// output matrix) should work as well.
///
/// (*) The conversion tested in this file unrolls along the reduction
/// dimension, which is not supported for scalable vectors.

#matmat_accesses_0 = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (m, n)>
]
#matmat_trait_0 = {
  indexing_maps = #matmat_accesses_0,
  iterator_types = ["parallel", "parallel", "reduction"]
}

#matmat_accesses_1 = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]
#matmat_trait_1 = {
  indexing_maps = #matmat_accesses_1,
  iterator_types = ["parallel", "parallel", "reduction"]
}

#matmat_accesses_2 = [
  affine_map<(m, n, k) -> (k, m)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (m, n)>
]
#matmat_trait_2 = {
  indexing_maps = #matmat_accesses_2,
  iterator_types = ["parallel", "parallel", "reduction"]
}

#matmat_accesses_3 = [
  affine_map<(m, n, k) -> (k, m)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]
#matmat_trait_3 = {
  indexing_maps = #matmat_accesses_3,
  iterator_types = ["parallel", "parallel", "reduction"]
}

#matmat_accesses_4 = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (n, m)>
]
#matmat_trait_4 = {
  indexing_maps = #matmat_accesses_4,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// ============================================================================
//  Matmul 0 (plain + masked + mixed types)
// ============================================================================
// CHECK-LABEL: func @matmul
// CHECK-SAME: %[[A:[a-zA-Z0-9]*]]: vector<2x4xf32>,
// CHECK-SAME: %[[B:[a-zA-Z0-9]*]]: vector<4x3xf32>,
// CHECK-SAME: %[[C:[a-zA-Z0-9]*]]: vector<2x3xf32>
//      CHECK: %[[At:.*]] = vector.transpose %[[A]], [1, 0]
// CHECK-SAME:  : vector<2x4xf32> to vector<4x2xf32>
//
//      CHECK: %[[a0:.*]] = vector.extract %[[At]][0] : vector<2xf32> from vector<4x2xf32>
//      CHECK: %[[b0:.*]] = vector.extract %[[B]][0] : vector<3xf32> from vector<4x3xf32>
//      CHECK: %[[c0:.*]] = vector.outerproduct %[[a0]], %[[b0]], %[[C]]
// CHECK-SAME:  : vector<2xf32>, vector<3xf32>
//
//      CHECK: %[[a1:.*]] = vector.extract %[[At]][1] : vector<2xf32> from vector<4x2xf32>
//      CHECK: %[[b1:.*]] = vector.extract %[[B]][1] : vector<3xf32> from vector<4x3xf32>
//      CHECK: %[[c1:.*]] = vector.outerproduct %[[a1]], %[[b1]], %[[c0]]
// CHECK-SAME:  : vector<2xf32>, vector<3xf32>
//
//      CHECK: %[[a2:.*]] = vector.extract %[[At]][2] : vector<2xf32> from vector<4x2xf32>
//      CHECK: %[[b2:.*]] = vector.extract %[[B]][2] : vector<3xf32> from vector<4x3xf32>
//      CHECK: %[[c2:.*]] = vector.outerproduct %[[a2]], %[[b2]], %[[c1]]
// CHECK-SAME:  : vector<2xf32>, vector<3xf32>
//
//      CHECK: %[[a3:.*]] = vector.extract %[[At]][3] : vector<2xf32> from vector<4x2xf32>
//      CHECK: %[[b3:.*]] = vector.extract %[[B]][3] : vector<3xf32> from vector<4x3xf32>
//      CHECK: %[[c3:.*]] = vector.outerproduct %[[a3]], %[[b3]], %[[c2]]
// CHECK-SAME:  : vector<2xf32>, vector<3xf32>
//
//      CHECK: return %[[c3]] : vector<2x3xf32>
func.func @matmul(%A: vector<2x4xf32>,
                  %B: vector<4x3xf32>,
                  %C: vector<2x3xf32>) -> vector<2x3xf32> {
  %0 = vector.contract #matmat_trait_0 %A, %B, %C
    : vector<2x4xf32>, vector<4x3xf32> into vector<2x3xf32>
  return %0 : vector<2x3xf32>
}

// CHECK-LABEL: func @matmul_scalable
// CHECK-SAME: %[[A:[a-zA-Z0-9]*]]: vector<2x4xf32>,
// CHECK-SAME: %[[B:[a-zA-Z0-9]*]]: vector<4x[3]xf32>,
// CHECK-SAME: %[[C:[a-zA-Z0-9]*]]: vector<2x[3]xf32>
//      CHECK: %[[At:.*]] = vector.transpose %[[A]], [1, 0]
// CHECK-SAME:  : vector<2x4xf32> to vector<4x2xf32>
//
//      CHECK: %[[a0:.*]] = vector.extract %[[At]][0] : vector<2xf32> from vector<4x2xf32>
//      CHECK: %[[b0:.*]] = vector.extract %[[B]][0] : vector<[3]xf32> from vector<4x[3]xf32>
//      CHECK: %[[c0:.*]] = vector.outerproduct %[[a0]], %[[b0]], %[[C]]
// CHECK-SAME:  : vector<2xf32>, vector<[3]xf32>
//
//      CHECK: %[[a1:.*]] = vector.extract %[[At]][1] : vector<2xf32> from vector<4x2xf32>
//      CHECK: %[[b1:.*]] = vector.extract %[[B]][1] : vector<[3]xf32> from vector<4x[3]xf32>
//      CHECK: %[[c1:.*]] = vector.outerproduct %[[a1]], %[[b1]], %[[c0]]
// CHECK-SAME:  : vector<2xf32>, vector<[3]xf32>
//
//      CHECK: %[[a2:.*]] = vector.extract %[[At]][2] : vector<2xf32> from vector<4x2xf32>
//      CHECK: %[[b2:.*]] = vector.extract %[[B]][2] : vector<[3]xf32> from vector<4x[3]xf32>
//      CHECK: %[[c2:.*]] = vector.outerproduct %[[a2]], %[[b2]], %[[c1]]
// CHECK-SAME:  : vector<2xf32>, vector<[3]xf32>
//
//      CHECK: %[[a3:.*]] = vector.extract %[[At]][3] : vector<2xf32> from vector<4x2xf32>
//      CHECK: %[[b3:.*]] = vector.extract %[[B]][3] : vector<[3]xf32> from vector<4x[3]xf32>
//      CHECK: %[[c3:.*]] = vector.outerproduct %[[a3]], %[[b3]], %[[c2]]
// CHECK-SAME:  : vector<2xf32>, vector<[3]xf32>
//
//      CHECK: return %[[c3]] : vector<2x[3]xf32>
func.func @matmul_scalable(%A: vector<2x4xf32>,
                           %B: vector<4x[3]xf32>,
                           %C: vector<2x[3]xf32>) -> vector<2x[3]xf32> {
  %0 = vector.contract #matmat_trait_0 %A, %B, %C
    : vector<2x4xf32>, vector<4x[3]xf32> into vector<2x[3]xf32>
  return %0 : vector<2x[3]xf32>
}

// CHECK-LABEL: func.func @masked_matmul(
// CHECK-SAME:    %{{.*}}: vector<3x5xf32>,
// CHECK-SAME:    %{{.*}}: vector<5x7xf32>,
// CHECK-SAME:    %{{.*}}: vector<3x7xf32>,
// CHECK-SAME:    %[[IN_MASK:.*]]: vector<3x7x5xi1>) -> vector<3x7xf32> {
// CHECK:         %[[T_MASK:.*]] = vector.transpose %[[IN_MASK]], [2, 0, 1] : vector<3x7x5xi1> to vector<5x3x7xi1>
// CHECK:         %[[T_MASK_R0:.*]] = vector.extract %[[T_MASK]][0] : vector<3x7xi1> from vector<5x3x7xi1>
// CHECK:         %{{.*}} = vector.mask %[[T_MASK_R0]] { vector.outerproduct %{{.*}} {kind = #vector.kind<add>} : vector<3xf32>, vector<7xf32> } : vector<3x7xi1> -> vector<3x7xf32>
// CHECK:         %[[T_MASK_R1:.*]] = vector.extract %[[T_MASK]][1] : vector<3x7xi1> from vector<5x3x7xi1>
// CHECK:         %{{.*}} = vector.mask %[[T_MASK_R1]] { vector.outerproduct %{{.*}} {kind = #vector.kind<add>} : vector<3xf32>, vector<7xf32> } : vector<3x7xi1> -> vector<3x7xf32>
// CHECK:         %[[T_MASK_R2:.*]] = vector.extract %[[T_MASK]][2] : vector<3x7xi1> from vector<5x3x7xi1>
// CHECK:         %{{.*}} = vector.mask %[[T_MASK_R2]] { vector.outerproduct %{{.*}} {kind = #vector.kind<add>} : vector<3xf32>, vector<7xf32> } : vector<3x7xi1> -> vector<3x7xf32>
// CHECK:         %[[T_MASK_R3:.*]] = vector.extract %[[T_MASK]][3] : vector<3x7xi1> from vector<5x3x7xi1>
// CHECK:         %{{.*}} = vector.mask %[[T_MASK_R3]] { vector.outerproduct %{{.*}} {kind = #vector.kind<add>} : vector<3xf32>, vector<7xf32> } : vector<3x7xi1> -> vector<3x7xf32>
// CHECK:         %[[T_MASK_R4:.*]] = vector.extract %[[T_MASK]][4] : vector<3x7xi1> from vector<5x3x7xi1>
// CHECK:         %{{.*}} = vector.mask %[[T_MASK_R4]] { vector.outerproduct %{{.*}} {kind = #vector.kind<add>} : vector<3xf32>, vector<7xf32> } : vector<3x7xi1> -> vector<3x7xf32>

func.func @masked_matmul(%A: vector<3x5xf32>,
                         %B: vector<5x7xf32>,
                         %C: vector<3x7xf32>,
                         %m : vector<3x7x5xi1>) -> vector<3x7xf32> {
  %0 = vector.mask %m { vector.contract #matmat_trait_0 %A, %B, %C
  : vector<3x5xf32>, vector<5x7xf32> into vector<3x7xf32> } : vector<3x7x5xi1> -> vector<3x7xf32>
  return %0 : vector<3x7xf32>
}

// CHECK-LABEL: func.func @masked_matmul_scalable(
// CHECK-SAME:    %{{.*}}: vector<3x5xf32>,
// CHECK-SAME:    %{{.*}}: vector<5x[7]xf32>,
// CHECK-SAME:    %{{.*}}: vector<3x[7]xf32>,
// CHECK-SAME:    %[[IN_MASK:.*]]: vector<3x[7]x5xi1>) -> vector<3x[7]xf32> {
// CHECK:         %[[T_MASK:.*]] = vector.transpose %[[IN_MASK]], [2, 0, 1] : vector<3x[7]x5xi1> to vector<5x3x[7]xi1>
// CHECK:         %[[T_MASK_R0:.*]] = vector.extract %[[T_MASK]][0] : vector<3x[7]xi1> from vector<5x3x[7]xi1>
// CHECK:         %{{.*}} = vector.mask %[[T_MASK_R0]] { vector.outerproduct %{{.*}} {kind = #vector.kind<add>} : vector<3xf32>, vector<[7]xf32> } : vector<3x[7]xi1> -> vector<3x[7]xf32>
// CHECK:         %[[T_MASK_R1:.*]] = vector.extract %[[T_MASK]][1] : vector<3x[7]xi1> from vector<5x3x[7]xi1>
// CHECK:         %[[VAL_13:.*]] = vector.mask %[[T_MASK_R1]] { vector.outerproduct %{{.*}} {kind = #vector.kind<add>} : vector<3xf32>, vector<[7]xf32> } : vector<3x[7]xi1> -> vector<3x[7]xf32>
// CHECK:         %[[T_MASK_R2:.*]] = vector.extract %[[T_MASK]][2] : vector<3x[7]xi1> from vector<5x3x[7]xi1>
// CHECK:         %{{.*}} = vector.mask %[[T_MASK_R2]] { vector.outerproduct %{{.*}} {kind = #vector.kind<add>} : vector<3xf32>, vector<[7]xf32> } : vector<3x[7]xi1> -> vector<3x[7]xf32>
// CHECK:         %[[T_MASK_R3:.*]] = vector.extract %[[T_MASK]][3] : vector<3x[7]xi1> from vector<5x3x[7]xi1>
// CHECK:         %{{.*}} = vector.mask %[[T_MASK_R3]] { vector.outerproduct %{{.*}} {kind = #vector.kind<add>} : vector<3xf32>, vector<[7]xf32> } : vector<3x[7]xi1> -> vector<3x[7]xf32>
// CHECK:         %[[T_MASK_R4:.*]] = vector.extract %[[T_MASK]][4] : vector<3x[7]xi1> from vector<5x3x[7]xi1>
// CHECK:         %{{.*}} = vector.mask %[[T_MASK_R4]] { vector.outerproduct %{{.*}} {kind = #vector.kind<add>} : vector<3xf32>, vector<[7]xf32> } : vector<3x[7]xi1> -> vector<3x[7]xf32>

func.func @masked_matmul_scalable(%A: vector<3x5xf32>,
                                  %B: vector<5x[7]xf32>,
                                  %C: vector<3x[7]xf32>,
                                  %m : vector<3x[7]x5xi1>) -> vector<3x[7]xf32> {
  %0 = vector.mask %m { vector.contract #matmat_trait_0 %A, %B, %C
  : vector<3x5xf32>, vector<5x[7]xf32> into vector<3x[7]xf32> } : vector<3x[7]x5xi1> -> vector<3x[7]xf32>
  return %0 : vector<3x[7]xf32>
}

// CHECK-LABEL: func @matmul_mixed
// CHECK-SAME: %[[A:[a-zA-Z0-9]*]]: vector<2x1xf16>,
// CHECK-SAME: %[[B:[a-zA-Z0-9]*]]: vector<1x3xf16>,
// CHECK-SAME: %[[C:[a-zA-Z0-9]*]]: vector<2x3xf32>
//      CHECK: %[[At:.*]] = vector.transpose %[[A]], [1, 0]
//      CHECK: %[[a0:.*]] = vector.extract %[[At]][0] : vector<2xf16> from vector<1x2xf16>
//      CHECK: %[[b0:.*]] = vector.extract %[[B]][0] : vector<3xf16> from vector<1x3xf16>
//      CHECK: %[[a1:.*]] = arith.extf %[[a0]] : vector<2xf16> to vector<2xf32>
//      CHECK: %[[b1:.*]] = arith.extf %[[b0]] : vector<3xf16> to vector<3xf32>
//      CHECK: %[[c0:.*]] = vector.outerproduct %[[a1]], %[[b1]], %[[C]]
//      CHECK: return %[[c0]] : vector<2x3xf32>
func.func @matmul_mixed(%A: vector<2x1xf16>,
                        %B: vector<1x3xf16>,
                        %C: vector<2x3xf32>) -> vector<2x3xf32>
{
  %0 = vector.contract #matmat_trait_0 %A, %B, %C
    : vector<2x1xf16>, vector<1x3xf16> into vector<2x3xf32>
  return %0 : vector<2x3xf32>
}

// CHECK-LABEL: func @matmul_mixed_scalable
// CHECK-SAME: %[[A:[a-zA-Z0-9]*]]: vector<2x1xf16>,
// CHECK-SAME: %[[B:[a-zA-Z0-9]*]]: vector<1x[3]xf16>,
// CHECK-SAME: %[[C:[a-zA-Z0-9]*]]: vector<2x[3]xf32>
//      CHECK: %[[At:.*]] = vector.transpose %[[A]], [1, 0]
//      CHECK: %[[a0:.*]] = vector.extract %[[At]][0] : vector<2xf16> from vector<1x2xf16>
//      CHECK: %[[b0:.*]] = vector.extract %[[B]][0] : vector<[3]xf16> from vector<1x[3]xf16>
//      CHECK: %[[a1:.*]] = arith.extf %[[a0]] : vector<2xf16> to vector<2xf32>
//      CHECK: %[[b1:.*]] = arith.extf %[[b0]] : vector<[3]xf16> to vector<[3]xf32>
//      CHECK: %[[c0:.*]] = vector.outerproduct %[[a1]], %[[b1]], %[[C]]
//      CHECK: return %[[c0]] : vector<2x[3]xf32>
func.func @matmul_mixed_scalable(%A: vector<2x1xf16>,
                                 %B: vector<1x[3]xf16>,
                                 %C: vector<2x[3]xf32>) -> vector<2x[3]xf32>
{
  %0 = vector.contract #matmat_trait_0 %A, %B, %C
    : vector<2x1xf16>, vector<1x[3]xf16> into vector<2x[3]xf32>
  return %0 : vector<2x[3]xf32>
}

// ============================================================================
//  Matmul 1 (plain + scalable)
// ============================================================================
// CHECK-LABEL: func @matmul_1
// CHECK-SAME: %[[A:[a-zA-Z0-9]*]]: vector<2x1xf32>,
// CHECK-SAME: %[[B:[a-zA-Z0-9]*]]: vector<3x1xf32>,
// CHECK-SAME: %[[C:[a-zA-Z0-9]*]]: vector<2x3xf32>
//      CHECK: %[[At:.*]] = vector.transpose %[[A]], [1, 0]
//      CHECK: %[[Bt:.*]] = vector.transpose %[[B]], [1, 0]
//      CHECK: %[[a0:.*]] = vector.extract %[[At]][0] : vector<2xf32> from vector<1x2xf32>
//      CHECK: %[[b0:.*]] = vector.extract %[[Bt]][0] : vector<3xf32> from vector<1x3xf32>
//      CHECK: %[[c0:.*]] = vector.outerproduct %[[a0]], %[[b0]], %[[C]]
//      CHECK: return %[[c0]] : vector<2x3xf32>
func.func @matmul_1(%A: vector<2x1xf32>,
                    %B: vector<3x1xf32>,
                    %C: vector<2x3xf32>) -> vector<2x3xf32>
{
  %0 = vector.contract #matmat_trait_1 %A, %B, %C
    : vector<2x1xf32>, vector<3x1xf32> into vector<2x3xf32>
  return %0 : vector<2x3xf32>
}

// CHECK-LABEL: func @matmul_1_scalable
// CHECK-SAME: %[[A:[a-zA-Z0-9]*]]: vector<2x1xf32>,
// CHECK-SAME: %[[B:[a-zA-Z0-9]*]]: vector<[3]x1xf32>,
// CHECK-SAME: %[[C:[a-zA-Z0-9]*]]: vector<2x[3]xf32>
//      CHECK: %[[At:.*]] = vector.transpose %[[A]], [1, 0]
//      CHECK: %[[Bt:.*]] = vector.transpose %[[B]], [1, 0]
//      CHECK: %[[a0:.*]] = vector.extract %[[At]][0] : vector<2xf32> from vector<1x2xf32>
//      CHECK: %[[b0:.*]] = vector.extract %[[Bt]][0] : vector<[3]xf32> from vector<1x[3]xf32>
//      CHECK: %[[c0:.*]] = vector.outerproduct %[[a0]], %[[b0]], %[[C]]
//      CHECK: return %[[c0]] : vector<2x[3]xf32>
func.func @matmul_1_scalable(%A: vector<2x1xf32>,
                             %B: vector<[3]x1xf32>,
                             %C: vector<2x[3]xf32>) -> vector<2x[3]xf32>
{
  %0 = vector.contract #matmat_trait_1 %A, %B, %C
    : vector<2x1xf32>, vector<[3]x1xf32> into vector<2x[3]xf32>
  return %0 : vector<2x[3]xf32>
}

// ============================================================================
//  Matmul 2 (plain + scalable)
// ============================================================================
// CHECK-LABEL: func @matmul_2
// CHECK-SAME: %[[A:[a-zA-Z0-9]*]]: vector<1x2xf32>,
// CHECK-SAME: %[[B:[a-zA-Z0-9]*]]: vector<1x3xf32>,
// CHECK-SAME: %[[C:[a-zA-Z0-9]*]]: vector<2x3xf32>
//      CHECK: %[[a0:.*]] = vector.extract %[[A]][0] : vector<2xf32> from vector<1x2xf32>
//      CHECK: %[[b0:.*]] = vector.extract %[[B]][0] : vector<3xf32> from vector<1x3xf32>
//      CHECK: %[[c0:.*]] = vector.outerproduct %[[a0]], %[[b0]], %[[C]]
//      CHECK: return %[[c0]] : vector<2x3xf32>
func.func @matmul_2(%A: vector<1x2xf32>,
                    %B: vector<1x3xf32>,
                    %C: vector<2x3xf32>) -> vector<2x3xf32>
{
  %0 = vector.contract #matmat_trait_2 %A, %B, %C
    : vector<1x2xf32>, vector<1x3xf32> into vector<2x3xf32>
  return %0 : vector<2x3xf32>
}

// CHECK-LABEL: func @matmul_2_scalable
// CHECK-SAME: %[[A:[a-zA-Z0-9]*]]: vector<1x2xf32>,
// CHECK-SAME: %[[B:[a-zA-Z0-9]*]]: vector<1x[3]xf32>,
// CHECK-SAME: %[[C:[a-zA-Z0-9]*]]: vector<2x[3]xf32>
//      CHECK: %[[a0:.*]] = vector.extract %[[A]][0] : vector<2xf32> from vector<1x2xf32>
//      CHECK: %[[b0:.*]] = vector.extract %[[B]][0] : vector<[3]xf32> from vector<1x[3]xf32>
//      CHECK: %[[c0:.*]] = vector.outerproduct %[[a0]], %[[b0]], %[[C]]
//      CHECK: return %[[c0]] : vector<2x[3]xf32>
func.func @matmul_2_scalable(%A: vector<1x2xf32>,
                             %B: vector<1x[3]xf32>,
                             %C: vector<2x[3]xf32>) -> vector<2x[3]xf32>
{
  %0 = vector.contract #matmat_trait_2 %A, %B, %C
    : vector<1x2xf32>, vector<1x[3]xf32> into vector<2x[3]xf32>
  return %0 : vector<2x[3]xf32>
}

// ============================================================================
//  Matmul 3 (plain + scalable)
// ============================================================================
// CHECK-LABEL: func @matmul_3
// CHECK-SAME: %[[A:[a-zA-Z0-9]*]]: vector<1x2xf32>,
// CHECK-SAME: %[[B:[a-zA-Z0-9]*]]: vector<3x1xf32>,
// CHECK-SAME: %[[C:[a-zA-Z0-9]*]]: vector<2x3xf32>
//      CHECK: %[[Bt:.*]] = vector.transpose %[[B]], [1, 0]
//      CHECK: %[[a0:.*]] = vector.extract %[[A]][0] : vector<2xf32> from vector<1x2xf32>
//      CHECK: %[[b0:.*]] = vector.extract %[[Bt]][0] : vector<3xf32> from vector<1x3xf32>
//      CHECK: %[[c0:.*]] = vector.outerproduct %[[a0]], %[[b0]], %[[C]]
//      CHECK: return %[[c0]] : vector<2x3xf32>
func.func @matmul_3(%A: vector<1x2xf32>,
                    %B: vector<3x1xf32>,
                    %C: vector<2x3xf32>) -> vector<2x3xf32>
{
  %0 = vector.contract #matmat_trait_3 %A, %B, %C
    : vector<1x2xf32>, vector<3x1xf32> into vector<2x3xf32>
  return %0 : vector<2x3xf32>
}

// CHECK-LABEL: func @matmul_3_scalable
// CHECK-SAME: %[[A:[a-zA-Z0-9]*]]: vector<1x2xf32>,
// CHECK-SAME: %[[B:[a-zA-Z0-9]*]]: vector<[3]x1xf32>,
// CHECK-SAME: %[[C:[a-zA-Z0-9]*]]: vector<2x[3]xf32>
//      CHECK: %[[Bt:.*]] = vector.transpose %[[B]], [1, 0]
//      CHECK: %[[a0:.*]] = vector.extract %[[A]][0] : vector<2xf32> from vector<1x2xf32>
//      CHECK: %[[b0:.*]] = vector.extract %[[Bt]][0] : vector<[3]xf32> from vector<1x[3]xf32>
//      CHECK: %[[c0:.*]] = vector.outerproduct %[[a0]], %[[b0]], %[[C]]
//      CHECK: return %[[c0]] : vector<2x[3]xf32>
func.func @matmul_3_scalable(%A: vector<1x2xf32>,
                             %B: vector<[3]x1xf32>,
                             %C: vector<2x[3]xf32>) -> vector<2x[3]xf32>
{
  %0 = vector.contract #matmat_trait_3 %A, %B, %C
    : vector<1x2xf32>, vector<[3]x1xf32> into vector<2x[3]xf32>
  return %0 : vector<2x[3]xf32>
}

// ============================================================================
//  Matmul 4 (plain + scalable)
// ============================================================================
// CHECK-LABEL: func @matmul_4
// CHECK-SAME: %[[A:[a-zA-Z0-9]*]]: vector<2x1xf32>,
// CHECK-SAME: %[[B:[a-zA-Z0-9]*]]: vector<1x3xf32>,
// CHECK-SAME: %[[C:[a-zA-Z0-9]*]]: vector<3x2xf32>
//      CHECK: %[[At:.*]] = vector.transpose %[[A]], [1, 0]
//      CHECK: %[[b0:.*]] = vector.extract %[[B]][0] : vector<3xf32> from vector<1x3xf32>
//      CHECK: %[[a0:.*]] = vector.extract %[[At]][0] : vector<2xf32> from vector<1x2xf32>
//      CHECK: %[[c0:.*]] = vector.outerproduct %[[b0]], %[[a0]], %[[C]]
//      CHECK: return %[[c0]] : vector<3x2xf32>
func.func @matmul_4(%A: vector<2x1xf32>,
                    %B: vector<1x3xf32>,
                    %C: vector<3x2xf32>) -> vector<3x2xf32>
{
  %0 = vector.contract #matmat_trait_4 %A, %B, %C
    : vector<2x1xf32>, vector<1x3xf32> into vector<3x2xf32>
  return %0 : vector<3x2xf32>
}

// CHECK-LABEL: func @matmul_4_scalable
// CHECK-SAME: %[[A:[a-zA-Z0-9]*]]: vector<[2]x1xf32>,
// CHECK-SAME: %[[B:[a-zA-Z0-9]*]]: vector<1x3xf32>,
// CHECK-SAME: %[[C:[a-zA-Z0-9]*]]: vector<3x[2]xf32>
//      CHECK: %[[At:.*]] = vector.transpose %[[A]], [1, 0]
//      CHECK: %[[b0:.*]] = vector.extract %[[B]][0] : vector<3xf32> from vector<1x3xf32>
//      CHECK: %[[a0:.*]] = vector.extract %[[At]][0] : vector<[2]xf32> from vector<1x[2]xf32>
//      CHECK: %[[c0:.*]] = vector.outerproduct %[[b0]], %[[a0]], %[[C]]
//      CHECK: return %[[c0]] : vector<3x[2]xf32>
func.func @matmul_4_scalable(%A: vector<[2]x1xf32>,
                             %B: vector<1x3xf32>,
                             %C: vector<3x[2]xf32>) -> vector<3x[2]xf32>
{
  %0 = vector.contract #matmat_trait_4 %A, %B, %C
    : vector<[2]x1xf32>, vector<1x3xf32> into vector<3x[2]xf32>
  return %0 : vector<3x[2]xf32>
}

// ============================================================================
//  TD sequence
// ============================================================================
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %module_op
      : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %f {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
    } : !transform.any_op
    transform.yield
  }
}
