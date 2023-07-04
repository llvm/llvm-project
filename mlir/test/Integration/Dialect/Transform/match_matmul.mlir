// RUN: mlir-opt %s --test-transform-dialect-interpreter --verify-diagnostics

module attributes { transform.with_named_sequence } {
  transform.named_sequence @match_matmul(%entry: !transform.any_op {transform.readonly})
      -> (!transform.any_op, !transform.any_op, !transform.param<i64>,
          !transform.type, !transform.type, !transform.type) {
    %c1 = transform.param.constant 1 : i64 -> !transform.param<i64>
    %c2 = transform.param.constant 2 : i64 -> !transform.param<i64>
    %capture:5 = transform.match.structured %entry : (!transform.any_op)
        -> (!transform.any_op, !transform.param<i64>, !transform.type, !transform.type, !transform.type) {
    ^bb0(%struct: !transform.any_op):
      transform.match.operation_name %struct ["linalg.matmul"] : !transform.any_op
      %dims = transform.match.structured.dim %struct[all] : (!transform.any_op) -> !transform.param<i64>
      
      %n_inputs = transform.match.structured.num_inputs %struct : (!transform.any_op) -> !transform.param<i64>
      %n_inits = transform.match.structured.num_inits %struct : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %n_inputs, %c2 : !transform.param<i64>
      transform.match.param.cmpi eq %n_inits, %c1 : !transform.param<i64>
      
      %lhs = transform.match.structured.input %struct[0] : (!transform.any_op) -> !transform.any_value
      %rhs = transform.match.structured.input %struct[1] : (!transform.any_op) -> !transform.any_value
      %res = transform.match.structured.result %struct[0] : (!transform.any_op) -> !transform.any_value
      %lhs_type = transform.get_type elemental %lhs : (!transform.any_value) -> !transform.type
      %rhs_type = transform.get_type elemental %rhs : (!transform.any_value) -> !transform.type
      %res_type = transform.get_type elemental %res : (!transform.any_value) -> !transform.type

      %init = transform.match.structured.init %struct[0] : (!transform.any_op) -> !transform.any_op
      transform.match.operation_name %init ["linalg.fill"] : !transform.any_op

      transform.match.structured.yield %init, %dims, %lhs_type, %rhs_type, %res_type
          : !transform.any_op, !transform.param<i64>, !transform.type, !transform.type, !transform.type
    }
    transform.yield %capture#0, %entry, %capture#1, %capture#2, %capture#3, %capture#4
        : !transform.any_op, !transform.any_op, !transform.param<i64>, !transform.type, !transform.type, !transform.type
  }

  transform.named_sequence @print_matmul(
      %fill: !transform.any_op {transform.readonly},
      %matmul: !transform.any_op {transform.readonly},
      %dims: !transform.param<i64> {transform.readonly},
      %lhs_type: !transform.type {transform.readonly},
      %rhs_type: !transform.type {transform.readonly},
      %res_type: !transform.type {transform.readonly}) {
    transform.test_print_remark_at_operand %fill, "fill" : !transform.any_op
    transform.test_print_remark_at_operand %matmul, "matmul" : !transform.any_op
    transform.test_print_param %dims, "dimensions" at %matmul : !transform.param<i64>, !transform.any_op
    transform.test_print_param %lhs_type, "LHS type" at %matmul : !transform.type, !transform.any_op
    transform.test_print_param %rhs_type, "RHS type" at %matmul : !transform.type, !transform.any_op
    transform.test_print_param %res_type, "result type" at %matmul : !transform.type, !transform.any_op
    transform.yield
  }

  transform.sequence failures(propagate) {
  ^bb(%root: !transform.any_op):
    foreach_match in %root
      @match_matmul -> @print_matmul
      : (!transform.any_op) -> !transform.any_op
  }
}

func.func @matmul_simple(%lhs: tensor<10x20xf16>, %rhs: tensor<20x15xf32>) -> tensor<10x15xf64>{
  %cst = arith.constant 0.0 : f64
  %empty = tensor.empty() : tensor<10x15xf64>
  // expected-remark @below {{fill}}
  %fill = linalg.fill ins(%cst : f64) outs(%empty : tensor<10x15xf64>) -> tensor<10x15xf64>
  // expected-remark @below {{matmul}}
  // expected-remark @below {{dimensions 10 : i64, 15 : i64, 20 : i64}}
  // expected-remark @below {{LHS type f16}}
  // expected-remark @below {{RHS type f32}}
  // expected-remark @below {{result type f64}}
  %result = linalg.matmul ins(%lhs, %rhs: tensor<10x20xf16>, tensor<20x15xf32>) outs(%fill: tensor<10x15xf64>) -> tensor<10x15xf64>
  return %result : tensor<10x15xf64>
}

func.func @matmul_with_extra_ops_in_func(%lhs: tensor<10x20xf32>, %rhs: tensor<20x15xf32>) -> tensor<10x15xf32> {
  %cst = arith.constant 0.0 : f64
  %empty = tensor.empty() : tensor<10x15xf32>

  // expected-remark @below {{fill}}
  %fill = linalg.fill ins(%cst : f64) outs(%empty : tensor<10x15xf32>) -> tensor<10x15xf32>

  %real_lhs = linalg.elemwise_binary { fun = #linalg.binary_fn<mul> } 
    ins(%lhs, %lhs : tensor<10x20xf32>, tensor<10x20xf32>) outs(%lhs : tensor<10x20xf32>) -> tensor<10x20xf32>

  // expected-remark @below {{matmul}}
  // expected-remark @below {{dimensions 10 : i64, 15 : i64, 20 : i64}}
  // expected-remark @below {{LHS type f32}}
  // expected-remark @below {{RHS type f32}}
  // expected-remark @below {{result type f32}}
  %result = linalg.matmul ins(%real_lhs, %rhs: tensor<10x20xf32>, tensor<20x15xf32>) outs(%fill: tensor<10x15xf32>) -> tensor<10x15xf32>
  return %result : tensor<10x15xf32>
}
