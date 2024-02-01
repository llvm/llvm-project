// RUN: transform-opt-ch4 %s --transform-interpreter --verify-diagnostics

// Matmul+ReLU.
func.func @fc_relu_operands_00(
    %lhs: tensor<512x512xf32>, %rhs: tensor<512x512xf32>,
    %bias: tensor<512x512xf32>, %output: tensor<512x512xf32>)
    -> tensor<512x512xf32> {
  // Matrix-matrix multiplication.
  // expected-remark @below {{matmul # 0}}
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
                          outs(%output: tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise addition.
  // expected-remark @below {{add # 0}}
  %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
    ins(%matmul, %bias : tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise max with 0 (ReLU).
  %c0f = arith.constant 0.0 : f32
  // expected-remark @below {{max # 0}}
  %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> }
    ins(%biased, %c0f : tensor<512x512xf32>, f32)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>
  func.return %relued : tensor<512x512xf32>
}

// Matmul+ReLU with swapped operands.
func.func @fc_relu_operands_01(
    %lhs: tensor<512x512xf32>, %rhs: tensor<512x512xf32>,
    %bias: tensor<512x512xf32>, %output: tensor<512x512xf32>)
    -> tensor<512x512xf32> {
  // Matrix-matrix multiplication.
  // expected-remark @below {{matmul # 1}}
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
                          outs(%output: tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise addition.
  // expected-remark @below {{add # 1}}
  %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
    ins(%matmul, %bias : tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise max with 0 (ReLU).
  %c0f = arith.constant 0.0 : f32
  // expected-remark @below {{max # 1}}
  %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> }
    ins(%c0f, %biased : f32, tensor<512x512xf32>)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>
  func.return %relued : tensor<512x512xf32>
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
      @match_matmul_elemwise -> @print_matmul_elemwise
      : (!transform.any_op) -> !transform.any_op

    transform.yield
  }

  // This is an action sequence.
  transform.named_sequence @print_matmul_elemwise(
      %matmul: !transform.any_op {transform.readonly},
      %add: !transform.any_op {transform.readonly},
      %max: !transform.any_op {transform.readonly},
      %pos: !transform.param<i32> {transform.readonly}) {
    transform.debug.emit_param_as_remark %pos, "matmul #" at %matmul
      : !transform.param<i32>, !transform.any_op
    transform.debug.emit_param_as_remark %pos, "add #" at %add
      : !transform.param<i32>, !transform.any_op
    transform.debug.emit_param_as_remark %pos, "max #" at %max
      : !transform.param<i32>, !transform.any_op
    transform.yield
  }

  // This is also a matcher sequence. It is similarly given an operation to
  // match and nested operations must succeed in order for a match to be deemed
  // successful. It starts matching from the last operation in the use-def chain
  // and goes back because each operand (use) has exactly one definition.
  transform.named_sequence @match_matmul_elemwise(
      %last: !transform.any_op {transform.readonly}) 
      -> (!transform.any_op, !transform.any_op, !transform.any_op,
          !transform.param<i32>) {
    // The last operation must be an elementwise binary.
    transform.match.operation_name %last ["linalg.elemwise_binary"]
      : !transform.any_op

    // One of its operands must be defined by another operation, to which we
    // will get a handle here. This is achieved thanks to a newly defined
    // operation that tries to match operands one by one using the match
    // operations nested in its region.
    %pos, %middle = transform.match.my.has_operand_satisfying %last
        : (!transform.any_op) -> (!transform.param<i32>, !transform.any_op) {
    ^bb0(%operand: !transform.any_value):
      // The operand must be defined by an operation.
      %def = transform.get_defining_op %operand 
        : (!transform.any_value) -> !transform.any_op
      // The defining operation must itself be an elementwise binary.
      transform.match.operation_name %def ["linalg.elemwise_binary"]
        : !transform.any_op
      transform.yield %def : !transform.any_op
    }
    
    // And the first operand of that operation must be defined by yet another
    // operation.
    %matmul = transform.get_producer_of_operand %middle[0]
      : (!transform.any_op) -> !transform.any_op
    // And that operation is a matmul.
    transform.match.operation_name %matmul ["linalg.matmul"] : !transform.any_op
    // We will yield the handles to the matmul and the two elementwise
    // operations separately. 
    transform.yield %matmul, %middle, %last, %pos
      : !transform.any_op, !transform.any_op, !transform.any_op,
        !transform.param<i32>
  }
}
