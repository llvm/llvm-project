// RUN: mlir-opt %s --transform-interpreter --split-input-file --verify-diagnostics

func.func @fusable_with_matching_offsets(%arg0: tensor<10x20xf32>, %arg1: tensor<10x20xf32>, %dest: tensor<100x200xf32>) -> tensor<100x200xf32> {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index

  %slice0 = tensor.insert_slice %arg0 into %dest[%c0, %c0] [10, 20] [1, 1] : tensor<10x20xf32> into tensor<100x200xf32>
  %slice1 = tensor.insert_slice %arg1 into %dest[%c0, %c0] [10, 20] [1, 1] : tensor<10x20xf32> into tensor<100x200xf32>

  // expected-remark @+1 {{can be fused with producer tensor.insert_slice ops}}
  %result = linalg.add ins(%slice0, %slice1 : tensor<100x200xf32>, tensor<100x200xf32>)
                       outs(%dest : tensor<100x200xf32>) -> tensor<100x200xf32>

  return %result : tensor<100x200xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %add = transform.structured.match ops{["linalg.add"]} in %arg : (!transform.any_op) -> !transform.any_op
    transform.test.query_producer_fusability %add : !transform.any_op
    transform.yield
  }
}

// -----

func.func @not_fusable_with_different_offsets(%arg0: tensor<10x20xf32>, %arg1: tensor<10x20xf32>, %dest: tensor<100x200xf32>) -> tensor<100x200xf32> {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index

  %slice0 = tensor.insert_slice %arg0 into %dest[%c0, %c0] [10, 20] [1, 1] : tensor<10x20xf32> into tensor<100x200xf32>
  %slice1 = tensor.insert_slice %arg1 into %dest[%c10, %c20] [10, 20] [1, 1] : tensor<10x20xf32> into tensor<100x200xf32>

  // expected-remark @+1 {{cannot be fused with producer tensor.insert_slice ops}}
  %result = linalg.add ins(%slice0, %slice1 : tensor<100x200xf32>, tensor<100x200xf32>)
                       outs(%dest : tensor<100x200xf32>) -> tensor<100x200xf32>

  return %result : tensor<100x200xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %add = transform.structured.match ops{["linalg.add"]} in %arg : (!transform.any_op) -> !transform.any_op
    transform.test.query_producer_fusability %add : !transform.any_op
    transform.yield
  }
}

// -----

func.func @fusable_with_consumer_extract_slice(%arg0: tensor<100x200xf32>, %arg1: tensor<100x200xf32>, %dest: tensor<100x200xf32>) -> tensor<10x20xf32> {
  // expected-remark @+1 {{can be fused with consumer tensor.extract_slice op}}
  %add = linalg.add ins(%arg0, %arg1 : tensor<100x200xf32>, tensor<100x200xf32>)
                    outs(%dest : tensor<100x200xf32>) -> tensor<100x200xf32>

  %c0 = arith.constant 0 : index
  %slice = tensor.extract_slice %add[%c0, %c0] [10, 20] [1, 1] : tensor<100x200xf32> to tensor<10x20xf32>

  return %slice : tensor<10x20xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %add = transform.structured.match ops{["linalg.add"]} in %arg : (!transform.any_op) -> !transform.any_op
    transform.test.query_consumer_fusability %add : !transform.any_op
    transform.yield
  }
}
