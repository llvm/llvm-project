// RUN: mlir-opt %s --transform-preload-library='transform-library-paths=%p/match_matmul_common.mlir' --transform-interpreter --verify-diagnostics

module attributes { transform.with_named_sequence } {
  transform.named_sequence @_match_matmul_like(
      %entry: !transform.any_op {transform.readonly},
      %rank: !transform.param<i64> {transform.readonly})
      -> (!transform.any_op, !transform.any_op, !transform.param<i64>,
          !transform.type, !transform.type, !transform.type,
          !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)

  transform.named_sequence @match_bmm(%entry: !transform.any_op {transform.readonly})
      -> (!transform.any_op, !transform.any_op, !transform.param<i64>,
          !transform.type, !transform.type, !transform.type, !transform.param<i64>) {
    transform.match.operation_name %entry ["linalg.batch_matmul", "linalg.generic"] : !transform.any_op
    %c3 = transform.param.constant 4 : i64 -> !transform.param<i64>
    %fill, %bmm, %dims, %lhs_type, %rhs_type, %res_type, %batch, %m, %n, %k =
      transform.include @_match_matmul_like failures(propagate) (%entry, %c3)
        : (!transform.any_op, !transform.param<i64>)
        -> (!transform.any_op, !transform.any_op, !transform.param<i64>,
            !transform.type, !transform.type, !transform.type,
            !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)

    transform.yield %fill, %bmm, %dims, %lhs_type, %rhs_type, %res_type, %batch
        : !transform.any_op, !transform.any_op, !transform.param<i64>, !transform.type, !transform.type, !transform.type, !transform.param<i64>
  }

  transform.named_sequence @print_bmm(
      %fill: !transform.any_op {transform.readonly},
      %bmm: !transform.any_op {transform.readonly},
      %dims: !transform.param<i64> {transform.readonly},
      %lhs_type: !transform.type {transform.readonly},
      %rhs_type: !transform.type {transform.readonly},
      %res_type: !transform.type {transform.readonly},
      %batch: !transform.param<i64> {transform.readonly}) {
    transform.debug.emit_remark_at %fill, "fill" : !transform.any_op
    transform.debug.emit_remark_at %bmm, "batch matmul" : !transform.any_op
    transform.debug.emit_param_as_remark %dims, "dimensions" at %bmm : !transform.param<i64>, !transform.any_op
    transform.debug.emit_param_as_remark %lhs_type, "LHS type" at %bmm : !transform.type, !transform.any_op
    transform.debug.emit_param_as_remark %rhs_type, "RHS type" at %bmm : !transform.type, !transform.any_op
    transform.debug.emit_param_as_remark %res_type, "result type" at %bmm : !transform.type, !transform.any_op
    transform.debug.emit_param_as_remark %batch, "batch dimension" at %bmm : !transform.param<i64>, !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.consumed}) {
    transform.foreach_match in %root
      @match_bmm -> @print_bmm
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

func.func @bmm_simple(%lhs: tensor<40x10x20xf16>, %rhs: tensor<40x20x15xf32>) -> tensor<40x10x15xf64>{
  %cst = arith.constant 0.0 : f64
  %empty = tensor.empty() : tensor<40x10x15xf64>
  // expected-remark @below {{fill}}
  %fill = linalg.fill ins(%cst : f64) outs(%empty : tensor<40x10x15xf64>) -> tensor<40x10x15xf64>
  // expected-remark @below {{batch matmul}}
  // expected-remark @below {{dimensions 40 : i64, 10 : i64, 15 : i64, 20 : i64}}
  // expected-remark @below {{LHS type f16}}
  // expected-remark @below {{RHS type f32}}
  // expected-remark @below {{result type f64}}
  // expected-remark @below {{batch dimension 0}}
  %result = linalg.batch_matmul ins(%lhs, %rhs: tensor<40x10x20xf16>, tensor<40x20x15xf32>) outs(%fill: tensor<40x10x15xf64>) -> tensor<40x10x15xf64>
  return %result : tensor<40x10x15xf64>
}
