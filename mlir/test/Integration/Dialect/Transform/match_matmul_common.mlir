// RUN: mlir-opt %s

module attributes { transform.with_named_sequence } {
  transform.named_sequence @_match_matmul_like(
      %entry: !transform.any_op {transform.readonly},
      %rank: !transform.param<i64> {transform.readonly})
      -> (!transform.any_op, !transform.any_op, !transform.param<i64>,
          !transform.type, !transform.type, !transform.type,
          !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>) {
    %c1 = transform.param.constant 1 : i64 -> !transform.param<i64>
    %c2 = transform.param.constant 2 : i64 -> !transform.param<i64>
    %capture:9 = transform.match.structured %entry : (!transform.any_op)
        -> (!transform.any_op, !transform.param<i64>, !transform.type, !transform.type, !transform.type,
            !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>) {
    ^bb0(%struct: !transform.any_op):
      %op_rank = transform.match.structured.rank %struct : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %op_rank : !transform.param<i64>
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

      transform.match.structured.body %struct { contraction = ["arith.mulf", "arith.addf"] } : !transform.any_op
      %dim_kinds:4 = transform.match.structured.classify_contraction_dims %struct
        : (!transform.any_op) -> (!transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)

      transform.match.structured.yield %init, %dims, %lhs_type, %rhs_type, %res_type, %dim_kinds#0, %dim_kinds#1, %dim_kinds#2, %dim_kinds#3
          : !transform.any_op, !transform.param<i64>, !transform.type, !transform.type, !transform.type,
            !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>
    }
    transform.yield %capture#0, %entry, %capture#1, %capture#2, %capture#3, %capture#4, 
                    %capture#5, %capture#6, %capture#7, %capture#8
        : !transform.any_op, !transform.any_op, !transform.param<i64>, !transform.type, !transform.type, !transform.type,
          !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>
  }
}
