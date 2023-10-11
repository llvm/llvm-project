// RUN: mlir-opt %s --test-transform-dialect-interpreter --verify-diagnostics --split-input-file

module attributes { transform.with_named_sequence } {
  transform.named_sequence @match_sparse_structured(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %0 = transform.match.structured %arg0 : (!transform.any_op) -> !transform.any_op {
    ^bb0(%struct: !transform.any_op):
      %sp_kernel = transform.sparse_tensor.match.sparse_inout %struct
          : (!transform.any_op) -> !transform.any_op
      transform.match.structured.yield %sp_kernel : !transform.any_op
    }
    transform.yield %0 : !transform.any_op
  }

  transform.named_sequence @print_sparse_structured(%arg0: !transform.any_op {transform.readonly}) {
    transform.test_print_remark_at_operand %arg0, "sparse_kernel" : !transform.any_op
    transform.yield
  }

  // Entry point. Match any structured sparse operation and emit at remark.
  transform.sequence failures(propagate) attributes { transform.target_tag = "transform" } {
  ^bb0(%arg0: !transform.any_op):
    transform.foreach_match in %arg0
        @match_sparse_structured -> @print_sparse_structured
        : (!transform.any_op) -> !transform.any_op
  }
}

#CSR = #sparse_tensor.encoding<{map = (d0, d1) -> (d0 : dense, d1 : compressed)}>

func.func @payload(%lhs: tensor<10x20xf16>,
                   %sp_lhs: tensor<10x20xf16, #CSR>,
                   %rhs: tensor<20x15xf32>) -> tensor<10x15xf64>{
  %cst = arith.constant 0.0 : f64
  %empty = tensor.empty() : tensor<10x15xf64>
  %fill = linalg.fill ins(%cst : f64) outs(%empty : tensor<10x15xf64>) -> tensor<10x15xf64>

  %result = linalg.matmul ins(%lhs, %rhs: tensor<10x20xf16>, tensor<20x15xf32>)
                         outs(%fill: tensor<10x15xf64>) -> tensor<10x15xf64>
  // expected-remark @below {{sparse_kernel}}
  %sp_in = linalg.matmul ins(%sp_lhs, %rhs: tensor<10x20xf16, #CSR>, tensor<20x15xf32>)
                        outs(%fill: tensor<10x15xf64>) -> tensor<10x15xf64>

  %sp_empty = tensor.empty() : tensor<10x15xf64, #CSR>
  // expected-remark @below {{sparse_kernel}}
  %sp_out = linalg.matmul ins(%lhs, %rhs: tensor<10x20xf16>, tensor<20x15xf32>)
                         outs(%sp_empty: tensor<10x15xf64, #CSR>) -> tensor<10x15xf64, #CSR>
  return %result : tensor<10x15xf64>
}
