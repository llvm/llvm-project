// RUN: mlir-opt %s -transform-interpreter -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @vectorize_matmul
// CHECK-SAME: %[[A:.*]]: tensor<24x12xf32>
// CHECK-SAME: %[[B:.*]]: tensor<12x25xf32>
// CHECK-SAME: %[[C:.*]]: tensor<24x25xf32>
func.func @vectorize_matmul(%arg0: tensor<24x12xf32>,
                            %arg1: tensor<12x25xf32>,
                            %arg2: tensor<24x25xf32>) -> tensor<24x25xf32> {
  // CHECK: %[[vA:.+]] = vector.transfer_read %[[A]]
  // CHECK: %[[vB:.+]] = vector.transfer_read %[[B]]
  // CHECK: %[[vC:.+]] = vector.transfer_read %[[C]]
  // CHECK: %[[vR:.+]] = vector.contract {{.*}} %[[vA]], %[[vB]], %[[vC]]
  // CHECK: vector.transfer_write %[[vR]], %[[C]]
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<24x12xf32>, tensor<12x25xf32>) outs(%arg2 : tensor<24x25xf32>) -> tensor<24x25xf32>
  func.return %0 : tensor<24x25xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @vectorize_matmul_memref
// CHECK-SAME: %[[A:.*]]: memref<24x12xf32>
// CHECK-SAME: %[[B:.*]]: memref<12x25xf32>
// CHECK-SAME: %[[C:.*]]: memref<24x25xf32>
func.func @vectorize_matmul_memref(%arg0: memref<24x12xf32>,
                                   %arg1: memref<12x25xf32>,
                                   %arg2: memref<24x25xf32>) {
  // CHECK: %[[vA:.+]] = vector.transfer_read %[[A]]
  // CHECK: %[[vB:.+]] = vector.transfer_read %[[B]]
  // CHECK: %[[vC:.+]] = vector.transfer_read %[[C]]
  // CHECK: %[[vR:.+]] = vector.contract {{.*}} %[[vA]], %[[vB]], %[[vC]]
  // CHECK: vector.transfer_write %[[vR]], %[[C]]
  linalg.matmul ins(%arg0, %arg1 : memref<24x12xf32>, memref<12x25xf32>) outs(%arg2 : memref<24x25xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @vectorize_copy_memref
// CHECK-SAME: %[[A:.*]]: memref<100x100xf32>,
// CHECK-SAME: %[[B:.*]]: memref<100x100xf32>
func.func @vectorize_copy_memref(%arg0: memref<100x100xf32>,
                                 %arg1: memref<100x100xf32>) {
  // CHECK: %[[vA:.+]] = vector.transfer_read %[[A]]
  // CHECK: vector.transfer_write %[[vA]], %[[B]]
  linalg.copy ins(%arg0 : memref<100x100xf32>) outs(%arg1 : memref<100x100xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.copy"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

#map0 = affine_map<()[s0] -> (-s0 + 12, 7)>
#map1 = affine_map<()[s0] -> (-s0 + 7)>

// CHECK-LABEL: @vectorize_keep_pad
// CHECK-SAME: %[[C:[a-zA-Z0-9_]+]]: tensor<24x25xf32>
func.func @vectorize_keep_pad(
    %arg0: tensor<24x12xf32>, %arg1: tensor<12x25xf32>,
    %arg2: tensor<24x25xf32>, %arg3: index, %arg4: index,
    %arg5: index) -> tensor<24x25xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = affine.min #map0()[%arg5]
  %1 = tensor.extract_slice %arg0[%arg3, %arg5] [4, %0] [1, 1] : tensor<24x12xf32> to tensor<4x?xf32>
  %2 = tensor.extract_slice %arg1[%arg5, %arg4] [%0, 5] [1, 1] : tensor<12x25xf32> to tensor<?x5xf32>
  %3 = tensor.extract_slice %arg2[%arg3, %arg4] [4, 5] [1, 1] : tensor<24x25xf32> to tensor<4x5xf32>
  %4 = affine.apply #map1()[%0]
  // CHECK: %[[pA:.*]] = tensor.pad
  %5 = tensor.pad %1 nofold low[%c0, %c0] high[%c0, %4] {
  ^bb0(%arg6: index, %arg7: index):
    tensor.yield %cst : f32
  } : tensor<4x?xf32> to tensor<4x7xf32>
  %6 = affine.apply #map1()[%0]
  // CHECK: %[[pB:.*]] = tensor.pad
  %7 = tensor.pad %2 nofold low[%c0, %c0] high[%6, %c0] {
  ^bb0(%arg6: index, %arg7: index):
    tensor.yield %cst : f32
  } : tensor<?x5xf32> to tensor<7x5xf32>
  // CHECK: %[[vA:.+]] = vector.transfer_read %[[pA]]
  // CHECK: %[[vB:.+]] = vector.transfer_read %[[pB]]
  // CHECK: %[[vC:.+]] = vector.transfer_read %[[C]]
  // CHECK: %[[vR:.+]] = vector.contract {{.*}} %[[vA]], %[[vB]], %[[vC]]
  // CHECK: vector.transfer_write %[[vR]], %[[C]]
  %8 = linalg.matmul ins(%5, %7 : tensor<4x7xf32>, tensor<7x5xf32>) outs(%3 : tensor<4x5xf32>) -> tensor<4x5xf32>
  %9 = tensor.insert_slice %8 into %arg2[%arg3, %arg4] [4, 5] [1, 1] : tensor<4x5xf32> into tensor<24x25xf32>
  return %9 : tensor<24x25xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

#map0 = affine_map<()[s0] -> (-s0 + 12, 7)>
#map1 = affine_map<()[s0] -> (-s0 + 7)>

// CHECK-LABEL: @vectorize_pad
// CHECK-SAME: %[[A:.+]]: tensor<24x12xf32>
// CHECK-SAME: %[[B:.+]]: tensor<12x25xf32>
// CHECK-SAME: %[[C:.+]]: tensor<24x25xf32>
func.func @vectorize_pad(
    %arg0: tensor<24x12xf32>, %arg1: tensor<12x25xf32>,
    %arg2: tensor<24x25xf32>, %arg3: index, %arg4: index,
    %arg5: index) -> tensor<24x25xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = affine.min #map0()[%arg5]
  // CHECK: %[[sA:.+]] = tensor.extract_slice %[[A]]
  // CHECK: %[[sB:.+]] = tensor.extract_slice %[[B]]
  %1 = tensor.extract_slice %arg0[%arg3, %arg5] [4, %0] [1, 1] : tensor<24x12xf32> to tensor<4x?xf32>
  %2 = tensor.extract_slice %arg1[%arg5, %arg4] [%0, 5] [1, 1] : tensor<12x25xf32> to tensor<?x5xf32>
  %3 = tensor.extract_slice %arg2[%arg3, %arg4] [4, 5] [1, 1] : tensor<24x25xf32> to tensor<4x5xf32>
  // CHECK: %[[vA:.+]] = vector.transfer_read %[[sA]]
  %4 = affine.apply #map1()[%0]
  %5 = tensor.pad %1 nofold low[%c0, %c0] high[%c0, %4] {
  ^bb0(%arg6: index, %arg7: index):
    tensor.yield %cst : f32
  } : tensor<4x?xf32> to tensor<4x7xf32>
  %6 = affine.apply #map1()[%0]
  // CHECK: %[[vB:.+]] = vector.transfer_read %[[sB]]
  %7 = tensor.pad %2 nofold low[%c0, %c0] high[%6, %c0] {
  ^bb0(%arg6: index, %arg7: index):
    tensor.yield %cst : f32
  } : tensor<?x5xf32> to tensor<7x5xf32>
  // CHECK: %[[vC:.+]] = vector.transfer_read %[[C]]
  // CHECK: %[[vR:.+]] = vector.contract {{.*}} %[[vA]], %[[vB]], %[[vC]]
  // CHECK: vector.transfer_write %[[vR]], %[[C]]
  %8 = linalg.matmul ins(%5, %7 : tensor<4x7xf32>, tensor<7x5xf32>) outs(%3 : tensor<4x5xf32>) -> tensor<4x5xf32>
  %9 = tensor.insert_slice %8 into %arg2[%arg3, %arg4] [4, 5] [1, 1] : tensor<4x5xf32> into tensor<24x25xf32>
  return %9 : tensor<24x25xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 {vectorize_padding} : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @vectorize(%arg0: tensor<24x12xf32>,
                     %arg1: tensor<12x25xf32>,
                     %arg2: tensor<24x25xf32>) -> tensor<24x25xf32> {
  // expected-note @below {{non-isolated target}}
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<24x12xf32>, tensor<12x25xf32>) outs(%arg2 : tensor<24x25xf32>) -> tensor<24x25xf32>
  func.return %0 : tensor<24x25xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{op requires isolated-from-above targets}}
    %2 = transform.structured.vectorize_children_and_apply_patterns %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Mixed Precision vetorization tests.

// CHECK-LABEL: func @mixed_precision_generic_as_contract
// CHECK-COUNT-3: vector.transfer_read
// CHECK-NOT:     arith.extf
// CHECK:         vector.contract
// CHECK:         vector.transfer_write
func.func @mixed_precision_generic_as_contract(%A: memref<8x16xbf16>, %B: memref<16x32xbf16>,
                         %C: memref<8x32xf32>) {
  linalg.generic {
    indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (k, n)>,
    affine_map<(m, n, k) -> (m, n)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  }
   ins(%A, %B : memref<8x16xbf16>, memref<16x32xbf16>)
   outs(%C : memref<8x32xf32>) {
    ^bb(%in: bf16, %in_0: bf16, %c: f32) :
      %a = arith.extf %in : bf16 to f32
      %b = arith.extf %in_0 : bf16 to f32
      %d = arith.mulf %a, %b: f32
      %e = arith.addf %c, %d: f32
      linalg.yield %e : f32
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1  { vectorize_mixed_precision, disable_transfer_permutation_map_lowering_patterns } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @mixed_precision_matmul_as_contract
// CHECK-COUNT-3: vector.transfer_read
// CHECK-NOT:     arith.extf
// CHECK:         vector.contract
// CHECK:         vector.transfer_write
func.func @mixed_precision_matmul_as_contract(%A: tensor<24x12xbf16>,
                              %B: tensor<12x25xbf16>,
                              %C: tensor<24x25xf32>) -> tensor<24x25xf32> {
  %0 = linalg.contract
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (k, n)>,
                       affine_map<(m, n, k) -> (m, n)>]
      ins(%A, %B : tensor<24x12xbf16>, tensor<12x25xbf16>)
      outs(%C : tensor<24x25xf32>) -> tensor<24x25xf32>
  func.return %0 : tensor<24x25xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.contract"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 { vectorize_mixed_precision } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @contraction_matmul
// CHECK-COUNT-3: vector.transfer_read
// CHECK-NOT:     arith.extf
// CHECK:         vector.contract
func.func @contraction_matmul(%A: memref<1584x1584xbf16>, %B: memref<1584x1584xbf16>, %C: memref<1584x1584xf32>) {
  linalg.matmul ins(%A, %B: memref<1584x1584xbf16>, memref<1584x1584xbf16>)
            outs(%C: memref<1584x1584xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1  { vectorize_mixed_precision } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
