// RUN: mlir-opt %s -test-transform-dialect-interpreter -split-input-file -verify-diagnostics | FileCheck %s

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

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @pdl_target : benefit(1) {
    %args = operands
    %results = types
    %0 = pdl.operation "linalg.matmul"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
    // TODO: we don't want this, but it is the required terminator for pdl.pattern
    rewrite %0 with "transform.dialect"
  }

  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
    %2 = transform.structured.vectorize %1
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

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
    %2 = transform.structured.vectorize %1
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

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
    %2 = transform.structured.vectorize %1 {vectorize_padding}
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

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    // expected-error @below {{op requires isolated-from-above targets}}
    %2 = transform.structured.vectorize %0
  }
}
