// RUN: mlir-opt %s --test-transform-dialect-interpreter='transform-library-paths=%p/match_matmul_common.mlir' --verify-diagnostics

module attributes { transform.with_named_sequence } {
  transform.named_sequence @_match_matmul_like(
      %entry: !transform.any_op {transform.readonly},
      %rank: !transform.param<i64> {transform.readonly})
      -> (!transform.any_op, !transform.any_op, !transform.param<i64>,
          !transform.type, !transform.type, !transform.type,
          !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)

  transform.named_sequence @match_matmul(%entry: !transform.any_op {transform.readonly})
      -> (!transform.any_op, !transform.any_op, !transform.param<i64>,
          !transform.type, !transform.type, !transform.type) {
    transform.match.operation_name %entry ["linalg.matmul", "linalg.generic"] : !transform.any_op
    %c3 = transform.param.constant 3 : i64 -> !transform.param<i64>
    %fill, %matmul, %dims, %lhs_type, %rhs_type, %res_type, %kinds:4 =
      transform.include @_match_matmul_like failures(propagate) (%entry, %c3)
        : (!transform.any_op, !transform.param<i64>)
        -> (!transform.any_op, !transform.any_op, !transform.param<i64>,
            !transform.type, !transform.type, !transform.type,
            !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)

    transform.yield %fill, %matmul, %dims, %lhs_type, %rhs_type, %res_type
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

func.func @matmul_generic(%lhs: tensor<10x20xf16>, %rhs: tensor<20x15xf32>) -> tensor<10x15xf64>{
  %cst = arith.constant 0.0 : f64
  %empty = tensor.empty() : tensor<10x15xf64>
  // expected-remark @below {{fill}}
  %fill = linalg.fill ins(%cst : f64) outs(%empty : tensor<10x15xf64>) -> tensor<10x15xf64>
  // expected-remark @below {{matmul}}
  // expected-remark @below {{dimensions 10 : i64, 15 : i64, 20 : i64}}
  // expected-remark @below {{LHS type f16}}
  // expected-remark @below {{RHS type f32}}
  // expected-remark @below {{result type f64}}
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%lhs, %rhs: tensor<10x20xf16>, tensor<20x15xf32>) outs(%fill: tensor<10x15xf64>) {
  ^bb(%arg0: f16, %arg1: f32, %arg2: f64):
    %0 = arith.extf %arg0 : f16 to f32
    %1 = arith.mulf %0, %arg1 : f32
    %2 = arith.extf %1 : f32 to f64
    %3 = arith.addf %2, %arg2 : f64
    linalg.yield %3 : f64
  }-> tensor<10x15xf64>
  return %result : tensor<10x15xf64>
}
