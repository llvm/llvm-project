// RUN:  mlir-opt %s -test-transform-dialect-interpreter --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @matmul_tensors_1(
func.func @matmul_tensors_1(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>,
  %arg2: tensor<128x128xf32> {linalg.inplaceable = true})
    -> tensor<128x128xf32> {
  // This operation is marked for tiling only.
  // CHECK-COUNT-3: scf.for
  // CHECK-COUNT-3: tensor.extract_slice
  // CHECK: linalg.matmul
  // CHECK-SAME: -> tensor<4x4xf32>
  %0 = linalg.matmul { test.attrA }
                      ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  func.return %0 : tensor<128x128xf32>
}

func.func @matmul_tensors_2(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>,
  %arg2: tensor<128x128xf32> {linalg.inplaceable = true})
    -> tensor<128x128xf32> {
  // This operation is marked f
  // This operation is marked for tiling and vectorization.
  // CHECK-COUNT-3: scf.for
  // CHECK-COUNT-3: vector.transfer_read
  // CHECK:       vector.contract
  // CHECK-NOT:   linalg.matmul
  // CHECK:       vector.transfer_write
  %0 = linalg.matmul { test.attrA, test.attrC }
                      ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  func.return %0 : tensor<128x128xf32>
}

func.func @matmul_tensors_3(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>,
  %arg2: tensor<128x128xf32> {linalg.inplaceable = true})
    -> tensor<128x128xf32> {
  // This operation is marked for vectorization only.
  // CHECK-NOT: scf.for
  // CHECK-COUNT-3: vector.transfer_read
  // CHECK: vector.contract
  // CHECK-SAME: into vector<128x128xf32>
  // CHECK: vector.transfer_write
  %0 = linalg.matmul { test.attrC }
                      ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  func.return %0 : tensor<128x128xf32>
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  // Match matmul operations inside @matmul_tensors with test.attrA set.
  pdl.pattern @pdl_target_attrA : benefit(1) {
    %args = operands
    %results = types
    %attr = attribute
    %0 = operation "linalg.matmul"(%args : !pdl.range<value>) {"test.attrA" = %attr}-> (%results : !pdl.range<type>)
    // TODO: we don't want this, but it is the required terminator for pdl.pattern
    rewrite %0 with "transform.dialect"
  }

  // Match matmul operations inside @matmul_tensors with test.attrC set.
  pdl.pattern @pdl_target_attrC : benefit(1) {
    %args = operands
    %results = types
    %attr = attribute
    %0 = operation "linalg.matmul"(%args : !pdl.range<value>) {"test.attrC" = %attr}-> (%results : !pdl.range<type>)
    // TODO: we don't want this, but it is the required terminator for pdl.pattern
    rewrite %0 with "transform.dialect"
  }

  transform.sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = pdl_match @pdl_target_attrA in %arg1
    transform.structured.tile %0 {sizes = [4, 4, 4]}
    %1 = pdl_match @pdl_target_attrC in %arg1
    %2 = transform.get_closest_isolated_parent %1
    transform.structured.vectorize %2
  }
}

// -----

// CHECK-LABEL: @vectorize_one
func.func @vectorize_one(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>,
  %arg2: tensor<128x128xf32> {linalg.inplaceable = true})
    -> tensor<128x128xf32> {
  // CHECK: vector.contract
  %0 = linalg.matmul {test.attrA}
                     ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  func.return %0 : tensor<128x128xf32>
}

func.func @vectorize_none(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>,
  %arg2: tensor<128x128xf32> {linalg.inplaceable = true})
    -> tensor<128x128xf32> {
  // CHECK: linalg.matmul
  %0 = linalg.matmul ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  func.return %0 : tensor<128x128xf32>
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @pdl_target : benefit(1) {
    %args = operands
    %results = types
    %attr = attribute
    %0 = operation "linalg.matmul"(%args : !pdl.range<value>) {"test.attrA" = %attr}-> (%results : !pdl.range<type>)
    // TODO: we don't want this, but it is the required terminator for pdl.pattern
    rewrite %0 with "transform.dialect"
  }

  transform.sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = pdl_match @pdl_target in %arg1
    %1 = get_closest_isolated_parent %0
    transform.structured.vectorize %1
  }
}

// -----

// CHECK-LABEL: @vectorize_all
func.func @vectorize_all(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>,
  %arg3: tensor<128x128xf32> {linalg.inplaceable = true})
    -> tensor<128x128xf32> {
  // CHECK: vector.contract
  %0 = linalg.matmul {test.attrA}
                     ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  // CHECK: vector.contract
  %1 = linalg.matmul ins(%arg0, %0: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg3: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  return %1 : tensor<128x128xf32>
}

transform.sequence {
^bb0(%arg0: !pdl.operation):
  transform.structured.vectorize %arg0
}
