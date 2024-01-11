// RUN: transform-opt-ch4 %s --transform-interpreter --verify-diagnostics

// Matmul as a named operation.
func.func @named(
    %lhs: tensor<512x512xf32>, %rhs: tensor<512x512xf32>,
    %bias: tensor<512x512xf32>, %output: tensor<512x512xf32>)
    -> tensor<512x512xf32> {
  // expected-remark @below {{matmul}}
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
                          outs(%output: tensor<512x512xf32>) -> tensor<512x512xf32>
  func.return %matmul : tensor<512x512xf32>
}

// Matmul as a generic operation.
func.func @generic(
    %lhs: tensor<512x512xf32>, %rhs: tensor<512x512xf32>,
    %bias: tensor<512x512xf32>, %output: tensor<512x512xf32>)
    -> tensor<512x512xf32> {
  // expected-remark @below {{matmul}}
  %matmul = linalg.generic {
    iterator_types = ["parallel", "parallel", "reduction"],
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d2, d1)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>]
  } ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%output: tensor<512x512xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %0 = arith.mulf %arg0, %arg1 : f32
    %1 = arith.addf %0, %arg2 : f32
    linalg.yield %1 : f32
  } -> tensor<512x512xf32>
  return %matmul : tensor<512x512xf32>
}

// The module containing named sequences must have an attribute allowing them
// to enable verification.
module @transforms attributes { transform.with_named_sequence } {
  // Entry point. This takes as the only argument the root operation (typically
  // pass root) given to the transform interpreter.
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.consumed}) {

    // Traverses the payload IR associated with the operand handle, invoking
    // @match_matmul_elemwise on each of the operations. If the named sequence
    // succeeds, i.e., if none of the nested match (transform) operations
    // produced a silenceable failure, invokes @print_matmul_elemwise and
    // forwards the values yielded as arguments of the new invocation. If the
    // named sequence fails with a silenceable failure, silences it (the message
    // is forwarded to the debug stream). Definite failures are propagated
    // immediately and unconditionally, as usual.
    transform.foreach_match in %root
      @match_generic_matmul -> @print_generic_matmul
      : (!transform.any_op) -> !transform.any_op

    transform.yield
  }

  // This is an action sequence.
  transform.named_sequence @print_generic_matmul(
      %matmul: !transform.any_op {transform.readonly}) {
    transform.test_print_remark_at_operand %matmul, "matmul" : !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_generic_matmul(
      %candidate: !transform.any_op {transform.readonly}) -> !transform.any_op {
    // Match a structured linear algebra operation.
    transform.match.structured %candidate : !transform.any_op {
    ^bb0(%c: !transform.any_op):
      // With a rank equal to 3.
      %rank = transform.match.structured.rank %c
        : (!transform.any_op) -> !transform.param<i64>
      %c3 = transform.param.constant 3 : i64 -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %c3 : !transform.param<i64>

      // With 2 inputs.
      %n_ins = transform.match.structured.num_inputs %c
        : (!transform.any_op) -> !transform.param<i64>
      %c2 = transform.param.constant 2 : i64 -> !transform.param<i64>
      transform.match.param.cmpi eq %n_ins, %c2 : !transform.param<i64>

      // With 1 output (note that structured ops in destination passing style
      // has as many inits as outputs).
      %n_inits = transform.match.structured.num_inits %c
        : (!transform.any_op) -> !transform.param<i64>
      %c1 = transform.param.constant 1 : i64 -> !transform.param<i64>
      transform.match.param.cmpi eq %n_inits, %c1 : !transform.param<i64>

      // All inputs and inits are accessed with a projected permutation.
      transform.match.structured.input %c[all] {projected_permutation}
        : !transform.any_op
      transform.match.structured.init %c[0] {projected_permutation}
        : !transform.any_op

      // The body is a mulf/addf contraction with appropriate dimensions.
      transform.match.structured.body %c 
        { contraction = ["arith.mulf", "arith.addf"] } : !transform.any_op
      %batch, %lhs, %rhs, %reduction =
      transform.match.structured.classify_contraction_dims %c
        : (!transform.any_op)
        -> (!transform.param<i64>, !transform.param<i64>, !transform.param<i64>,
            !transform.param<i64>)

      // There is one of lhs, rhs and reduction dimensions and zero batch
      // dimensions.
      %n_batch = transform.num_associations %batch
        : (!transform.param<i64>) -> !transform.param<i64>
      %n_lhs = transform.num_associations %lhs
        : (!transform.param<i64>) -> !transform.param<i64>
      %n_rhs = transform.num_associations %rhs
        : (!transform.param<i64>) -> !transform.param<i64>
      %n_reduction = transform.num_associations %reduction
        : (!transform.param<i64>) -> !transform.param<i64>
      %c0 = transform.param.constant 0 : i64 -> !transform.param<i64>
      transform.match.param.cmpi eq %n_batch, %c0 : !transform.param<i64>
      transform.match.param.cmpi eq %n_lhs, %c1 : !transform.param<i64>
      transform.match.param.cmpi eq %n_rhs, %c1 : !transform.param<i64>
      transform.match.param.cmpi eq %n_reduction, %c1 : !transform.param<i64>
    }
    transform.yield %candidate : !transform.any_op
  }
}
