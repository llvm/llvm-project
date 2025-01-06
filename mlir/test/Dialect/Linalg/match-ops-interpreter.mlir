// RUN: mlir-opt %s --pass-pipeline="builtin.module(transform-interpreter{debug-payload-root-tag=start_here})" --split-input-file --verify-diagnostics

module attributes { transform.with_named_sequence } {
  transform.named_sequence @print_structured(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "structured" : !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_structured_empty(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %0 = transform.match.structured %arg0 : (!transform.any_op) -> !transform.any_op {
    ^bb0(%arg1: !transform.any_op):
          transform.match.structured.yield %arg1 : !transform.any_op
    }
    transform.yield %0 : !transform.any_op
  }

  // Entry point. Match any structured operation and emit at remark.
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
    transform.foreach_match in %arg0
        @match_structured_empty -> @print_structured
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @payload() attributes { transform.target_tag = "start_here" } {
    %preA = tensor.empty() : tensor<2x3xf32>
    %cA = arith.constant 1.0 : f32
    // expected-remark @below {{structured}}
    %A = linalg.fill ins(%cA : f32) outs(%preA : tensor<2x3xf32>) -> tensor<2x3xf32>

    %B = arith.constant dense<1.0> : tensor<3x4xf32>
    %C = arith.constant dense<1000.0> : tensor<2x4xf32>
    // expected-remark @below {{structured}}
    %D = linalg.matmul ins(%A, %B: tensor<2x3xf32>, tensor<3x4xf32>)
                       outs(%C: tensor<2x4xf32>) -> tensor<2x4xf32>

    %E = arith.constant dense<2.0> : tensor<2x4xf32>
    // expected-remark @below {{structured}}
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%D : tensor<2x4xf32>) outs(%E : tensor<2x4xf32>) {
    ^bb0(%arg0: f32, %arg1: f32):
      linalg.yield %arg0 : f32
    } -> tensor<2x4xf32>

    return
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @do_nothing(%arg0: !transform.any_op {transform.readonly}) {
    transform.yield
  }

  transform.named_sequence @print_in_matcher(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.print %arg0 : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
    transform.foreach_match in %arg0
        @print_in_matcher -> @do_nothing
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @payload() attributes { transform.target_tag = "start_here" } {
    // CHECK: [[ IR Printer ]]
    // CHECK: test.print_me
    %0 = "test.print_me"() : () -> (i1)
    return
  }
}

// -----


module attributes { transform.with_named_sequence } {
  transform.named_sequence @do_nothing(%arg0: !transform.any_op {transform.readonly}) {
    transform.yield
  }

  // Entry point. Match any structured operation and emit a remark. Also emit
  // a different remark at all considered operations. When it fails, the
  // failure is suppressed and the resulting handle is assocaited with an empty
  // list, hence nothing is printed. Both remark printing operations happen
  // after the check in the sequence, so they only apply if the check operation
  // produced success (due to failure suppression or not).
  transform.named_sequence @match_structured_suppress(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %0 = transform.match.structured failures(suppress) %arg0 : (!transform.any_op) -> !transform.any_op {
    ^bb0(%arg1: !transform.any_op):
      transform.match.structured.yield %arg1 : !transform.any_op
    }
    transform.debug.emit_remark_at %0, "structured" : !transform.any_op
    transform.debug.emit_remark_at %arg0, "other" : !transform.any_op
    transform.yield %0 : !transform.any_op
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
    transform.foreach_match restrict_root in %arg0
        @match_structured_suppress -> @do_nothing
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  // expected-remark @below {{other}}
  func.func @payload() attributes { transform.target_tag = "start_here" } {
    // expected-remark @below {{other}}
    %D = arith.constant dense<1.0> : tensor<2x4xf32>
    // expected-remark @below {{other}}
    %E = arith.constant dense<2.0> : tensor<2x4xf32>
    // expected-remark @below {{structured}}
    // expected-remark @below {{other}}
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%D : tensor<2x4xf32>) outs(%E : tensor<2x4xf32>) {
    ^bb0(%arg0: f32, %arg1: f32):
      // expected-remark @below {{other}}
      linalg.yield %arg0 : f32
    } -> tensor<2x4xf32>

    // expected-remark @below {{other}}
    return
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @print_passthrough(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "passthrough" : !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_structured_body_passthrough(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %0 = transform.match.structured failures(propagate) %arg0 : (!transform.any_op) -> !transform.any_op {
    ^bb0(%arg1: !transform.any_op):
      transform.match.structured.body %arg1 { passthrough } : !transform.any_op
      transform.match.structured.yield %arg1 : !transform.any_op
    }
    transform.yield %0 : !transform.any_op
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
    transform.foreach_match in %arg0
        @match_structured_body_passthrough -> @print_passthrough
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @payload(%in: tensor<2xf32>, %out: tensor<2xf32>) attributes { transform.target_tag = "start_here" } {
    // expected-remark @below {{passthrough}}
    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%in : tensor<2xf32>) outs(%out : tensor<2xf32>) {
    ^bb0(%arg0: f32, %arg1: f32):
      linalg.yield %arg0 : f32
    } -> tensor<2xf32>

    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%in : tensor<2xf32>) outs(%out : tensor<2xf32>) {
    ^bb0(%arg0: f32, %arg1: f32):
      %0 = arith.mulf %arg0, %arg1 : f32
      linalg.yield %0 : f32
    } -> tensor<2xf32>

    // expected-remark @below {{passthrough}}
    linalg.copy ins(%in : tensor<2xf32>) outs(%out : tensor<2xf32>) -> tensor<2xf32>

    return
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @print_elementwise(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "elementwise" : !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_structured_body_elementwise(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %0 = transform.match.structured failures(propagate) %arg0 : (!transform.any_op) -> !transform.any_op {
    ^bb0(%arg1: !transform.any_op):
      transform.match.structured.body %arg1 { elementwise } : !transform.any_op
      transform.match.structured.yield %arg1 : !transform.any_op
    }
    transform.yield %0 : !transform.any_op
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
    transform.foreach_match in %arg0
        @match_structured_body_elementwise -> @print_elementwise
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @payload(%in1: tensor<2xf32>, %in2: tensor<2xf32>, %in3: tensor<2x3xf32>, %out: tensor<2xf32>, %out2: tensor<2x3xf32>) -> (tensor<2xf32>, tensor<2x3xf32>, tensor<2x3xf32>) attributes { transform.target_tag = "start_here" } {
    %cst0 = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // expected-remark @below {{elementwise}}
    %fill = linalg.fill ins(%cst0: f32) outs(%out: tensor<2xf32>) -> tensor<2xf32>
    // expected-remark @below {{elementwise}}
    %add = linalg.map {arith.addf} ins(%in1, %in2: tensor<2xf32>, tensor<2xf32>) outs(%fill: tensor<2xf32>)
    %non_elementwise = linalg.generic
      {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]}
      ins(%in1, %in3: tensor<2xf32>, tensor<2x3xf32>) outs(%out2: tensor<2x3xf32>) {
        ^bb0(%arg0: f32, %arg1: f32, %arg3: f32):
          %0 = arith.addf %arg0, %arg1 : f32
          %1 = tensor.dim %add, %c0 : tensor<2xf32>
          %2 = arith.subi %1, %c1 : index
          %3 = tensor.extract %add[%2] : tensor<2xf32>
          %4 = arith.mulf %0, %3 : f32
          linalg.yield %4 : f32
      } -> tensor<2x3xf32>
    // expected-remark @below {{elementwise}}
    %add_bcast = linalg.generic
      {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]}
      ins(%in1, %in3: tensor<2xf32>, tensor<2x3xf32>) outs(%out2: tensor<2x3xf32>) {
        ^bb0(%arg0: f32, %arg1: f32, %arg3: f32):
          %0 = arith.addf %arg0, %arg1 : f32
          linalg.yield %0 : f32
      } -> tensor<2x3xf32>
    return %add, %add_bcast, %non_elementwise : tensor<2xf32>, tensor<2x3xf32>, tensor<2x3xf32>
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @print_reduction(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "reduction" : !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_structured_body_reduction(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %0 = transform.match.structured failures(propagate) %arg0 : (!transform.any_op) -> !transform.any_op {
    ^bb0(%arg1: !transform.any_op):
      transform.match.structured.body %arg1 { reduction_position = 0 } : !transform.any_op
      transform.match.structured.yield %arg1 : !transform.any_op
    }
    transform.yield %0 : !transform.any_op
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
    transform.foreach_match in %arg0
        @match_structured_body_reduction -> @print_reduction
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @payload(%lhs: tensor<2x4xf32>, %rhs: tensor<4x3xf32>, %out: tensor<2x3xf32>) attributes { transform.target_tag = "start_here" } {
    // expected-remark @below {{reduction}}
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%lhs, %rhs: tensor<2x4xf32>, tensor<4x3xf32>) outs(%out: tensor<2x3xf32>) {
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
      %0 = arith.mulf %arg0, %arg1 : f32
      %1 = arith.addf %0, %arg2 : f32
      linalg.yield %1 : f32
    } -> tensor<2x3xf32>

    %r = tensor.empty() : tensor<2x3xf32>
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%lhs, %rhs: tensor<2x4xf32>, tensor<4x3xf32>) outs(%out, %r: tensor<2x3xf32>, tensor<2x3xf32>) {
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32):
      %0 = arith.mulf %arg0, %arg1 : f32
      %1 = arith.cmpf olt, %0, %arg2 : f32
      %2 = arith.select %1, %0, %arg2 : f32
      %3 = arith.select %1, %arg3, %0 : f32
      linalg.yield %2, %3 : f32, f32
    } -> (tensor<2x3xf32>, tensor<2x3xf32>)

    // expected-remark @below {{reduction}}
    linalg.matmul ins(%lhs, %rhs: tensor<2x4xf32>, tensor<4x3xf32>) outs(%out: tensor<2x3xf32>) -> tensor<2x3xf32>

    %e = tensor.empty() : tensor<2x4xf32>
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%lhs: tensor<2x4xf32>) outs(%e: tensor<2x4xf32>) {
    ^bb0(%arg0: f32, %arg1: f32):
      linalg.yield %arg0 : f32
    } -> tensor<2x4xf32>

    return
  }
}


// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @do_nothing(%arg0: !transform.any_op {transform.readonly}) {
    transform.yield
  }

  transform.named_sequence @print_dimension_size_match(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "matched sizes" : !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_dimension_capture(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    // Capture multiple dimension values. Suppress failures so we can print them anyway after the capture.
    %0:9 = transform.match.structured failures(suppress) %arg0
      : (!transform.any_op) -> (!transform.any_op, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>,
            !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>) {
    ^bb0(%arg1: !transform.any_op):
      // This also tests the positional specification used by other ops, which may not test it again.
      %1 = transform.match.structured.dim %arg1[all] : (!transform.any_op) -> !transform.param<i64>
      %2 = transform.match.structured.dim %arg1[0] : (!transform.any_op) -> !transform.param<i64>
      %3 = transform.match.structured.dim %arg1[-1] : (!transform.any_op) -> !transform.param<i64>
      %4 = transform.match.structured.dim %arg1[0, 2] : (!transform.any_op) -> !transform.param<i64>
      %5 = transform.match.structured.dim %arg1[0, -1] : (!transform.any_op) -> !transform.param<i64>
      %6 = transform.match.structured.dim %arg1[except(-1)] : (!transform.any_op) -> !transform.param<i64>
      %7 = transform.match.structured.dim %arg1[except(0, -2)] : (!transform.any_op) -> !transform.param<i64>
      %8 = transform.match.structured.dim %arg1[0, -3] : (!transform.any_op) -> !transform.param<i64>
      transform.match.structured.yield %arg1, %1, %2, %3, %4, %5, %6, %7, %8
          : !transform.any_op, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>,
            !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>
    }
    transform.debug.emit_param_as_remark %0#1, "dimensions all:" at %0#0 : !transform.param<i64>, !transform.any_op
    transform.debug.emit_param_as_remark %0#2, "dimension 0:" at %0#0 : !transform.param<i64>, !transform.any_op
    transform.debug.emit_param_as_remark %0#3, "dimension -1:" at %0#0 : !transform.param<i64>, !transform.any_op
    transform.debug.emit_param_as_remark %0#4, "dimensions 0, 2:" at %0#0 : !transform.param<i64>, !transform.any_op
    transform.debug.emit_param_as_remark %0#5, "dimensions 0, -1:" at %0#0 : !transform.param<i64>, !transform.any_op
    transform.debug.emit_param_as_remark %0#6, "dimensions except -1:" at %0#0 : !transform.param<i64>, !transform.any_op
    transform.debug.emit_param_as_remark %0#7, "dimensions except 0, -2:" at %0#0 : !transform.param<i64>, !transform.any_op
    transform.debug.emit_param_as_remark %0#8, "dimensions 0, -3:" at %0#0 : !transform.param<i64>, !transform.any_op
    transform.yield %0#0 : !transform.any_op
  }

  transform.named_sequence @match_dimension_sizes(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    %0 = transform.match.structured failures(propagate) %arg0 : (!transform.any_op) -> !transform.any_op {
    ^bb0(%arg1: !transform.any_op):
      %1 = transform.match.structured.dim %arg1[all] : (!transform.any_op) -> !transform.param<i64>
      %c2 = transform.param.constant 2 : i64 -> !transform.param<i64>
      %c3 = transform.param.constant 3 : i64 -> !transform.param<i64>
      %c4 = transform.param.constant 4 : i64 -> !transform.param<i64>
      %2 = transform.merge_handles %c2, %c3, %c4 : !transform.param<i64>
      transform.match.param.cmpi eq %1, %2 : !transform.param<i64>

      transform.match.structured.yield %arg1 : !transform.any_op
    }
    transform.yield %0 : !transform.any_op
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
    %0 = transform.foreach_match in %arg0 @match_dimension_capture -> @do_nothing : (!transform.any_op) -> !transform.any_op
    %1 = transform.foreach_match in %0 @match_dimension_sizes -> @print_dimension_size_match : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @payload(%lhs: tensor<2x4xf32>, %rhs: tensor<4x3xf32>, %out: tensor<2x3xf32>) attributes { transform.target_tag = "start_here" } {
    // The last does not emit anything because it fails to match
    // due to 0 and -3 being the same dimension in the 3D case.
    // expected-remark @below {{dimensions all: 2 : i64, 3 : i64, 4 : i64}}
    // expected-remark @below {{dimension 0: 2 : i64}}
    // expected-remark @below {{dimension -1: 4 : i64}}
    // expected-remark @below {{dimensions 0, 2: 2 : i64, 4 : i64}}
    // expected-remark @below {{dimensions 0, -1: 2 : i64, 4 : i64}}
    // expected-remark @below {{dimensions except -1: 2 : i64, 3 : i64}}
    // expected-remark @below {{dimensions except 0, -2: 4 : i64}}
    // expected-remark @below {{dimensions 0, -3:}}
    // expected-remark @below {{matched sizes}}
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%lhs, %rhs: tensor<2x4xf32>, tensor<4x3xf32>) outs(%out: tensor<2x3xf32>) {
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
      %0 = arith.mulf %arg0, %arg1 : f32
      %1 = arith.addf %0, %arg2 : f32
      linalg.yield %1 : f32
    } -> tensor<2x3xf32>

    return
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @print_all_reduction(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "all reduction" : !transform.any_op
    transform.yield
  }
  transform.named_sequence @print_all_parallel(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "all parallel" : !transform.any_op
    transform.yield
  }
  transform.named_sequence @print_last_reduction(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "last reduction" : !transform.any_op
    transform.yield
  }
  transform.named_sequence @print_parallel_except_last(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "parallel except last" : !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_all_reduction(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.structured failures(propagate) %arg0 : !transform.any_op {
    ^bb0(%arg1: !transform.any_op):
      transform.match.structured.dim %arg1[all] { reduction } : !transform.any_op
      transform.match.structured.yield
    }
    transform.yield %arg0 : !transform.any_op
  }
  transform.named_sequence @match_all_parallel(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.structured failures(propagate) %arg0 : !transform.any_op {
    ^bb0(%arg1: !transform.any_op):
      transform.match.structured.dim %arg1[all] { parallel } : !transform.any_op
      transform.match.structured.yield
    }
    transform.yield %arg0 : !transform.any_op
  }
  transform.named_sequence @match_last_reduction(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.structured failures(propagate) %arg0 : !transform.any_op {
    ^bb0(%arg1: !transform.any_op):
      transform.match.structured.dim %arg1[-1] { reduction } : !transform.any_op
      transform.match.structured.yield
    }
    transform.yield %arg0 : !transform.any_op
  }
  transform.named_sequence @match_parallel_except_last(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.structured failures(propagate) %arg0 : !transform.any_op {
    ^bb0(%arg1: !transform.any_op):
      transform.match.structured.dim %arg1[except(-1)] { parallel } : !transform.any_op
      transform.match.structured.yield
    }
    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
    %0 = transform.foreach_match in %arg0 @match_all_reduction -> @print_all_reduction : (!transform.any_op) -> !transform.any_op
    %1 = transform.foreach_match in %0 @match_all_parallel -> @print_all_parallel : (!transform.any_op) -> !transform.any_op
    %2 = transform.foreach_match in %1 @match_last_reduction -> @print_last_reduction : (!transform.any_op) -> !transform.any_op
    %3 = transform.foreach_match in %2 @match_parallel_except_last -> @print_parallel_except_last : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @payload(%lhs: tensor<2x4xf32>, %rhs: tensor<4x3xf32>, %out: tensor<2x3xf32>) attributes { transform.target_tag = "start_here" } {
    // expected-remark @below {{last reduction}}
    // expected-remark @below {{parallel except last}}
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%lhs, %rhs: tensor<2x4xf32>, tensor<4x3xf32>) outs(%out: tensor<2x3xf32>) {
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
      %0 = arith.mulf %arg0, %arg1 : f32
      %1 = arith.addf %0, %arg2 : f32
      linalg.yield %1 : f32
    } -> tensor<2x3xf32>

    // expected-remark @below {{last reduction}}
    // expected-remark @below {{parallel except last}}
    linalg.matmul ins(%lhs, %rhs : tensor<2x4xf32>, tensor<4x3xf32>) outs(%out : tensor<2x3xf32>) -> tensor<2x3xf32>

    %cst = arith.constant 1.0 : f32
    // expected-remark @below {{all parallel}}
    // expected-remark @below {{parallel except last}}
    linalg.fill ins(%cst : f32) outs(%out: tensor<2x3xf32>) -> tensor<2x3xf32>

    return
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @match_bitwidth(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.param<i64>) {
    %bw = transform.match.structured failures(propagate) %arg0 : (!transform.any_op) -> !transform.param<i64> {
    ^bb0(%arg1: !transform.any_op):
      %0 = transform.match.structured.init %arg1 [0] : (!transform.any_op) -> !transform.any_value
      %1 = transform.match.structured.elemental_bitwidth %0 : (!transform.any_value) -> !transform.param<i64>
      transform.match.structured.yield %1 : !transform.param<i64>
    }
    transform.yield %arg0, %bw : !transform.any_op, !transform.param<i64>
  }

  transform.named_sequence @print_bitwidth(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.param<i64> {transform.readonly}) {
    transform.debug.emit_param_as_remark %arg1, "bitwidth:" at %arg0 : !transform.param<i64>, !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
    transform.foreach_match in %arg0 @match_bitwidth -> @print_bitwidth : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @payload(%f32: f32, %tf32: tensor<?xf32>,
                     %index: index, %tindex: tensor<?xindex>)
            attributes { transform.target_tag = "start_here" }  {
    // expected-remark @below {{bitwidth: 32}}
    linalg.fill ins(%f32: f32) outs(%tf32: tensor<?xf32>) -> tensor<?xf32>
    linalg.fill ins(%index: index) outs(%tindex: tensor<?xindex>) -> tensor<?xindex>
    return
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @match_init(%arg0: !transform.any_op {transform.readonly})
      -> (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) {
    %outs:3 = transform.match.structured failures(suppress) %arg0
      : (!transform.any_op) -> (!transform.any_value, !transform.any_value, !transform.any_op) {
    ^bb0(%arg1: !transform.any_op):
      %0 = transform.match.structured.init %arg1 [0] : (!transform.any_op) -> !transform.any_value
      %1 = transform.match.structured.init %arg1 [all] : (!transform.any_op) -> !transform.any_value
      %2 = transform.match.structured.init %arg1 [0] : (!transform.any_op) -> !transform.any_op
      transform.match.structured.yield %0, %1, %2 : !transform.any_value, !transform.any_value, !transform.any_op
    }
    transform.yield %arg0, %outs#0, %outs#1, %outs#2 : !transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op
  }

  transform.named_sequence @print_init(%arg0: !transform.any_op {transform.readonly},
                                         %arg1: !transform.any_value {transform.readonly},
                                         %arg2: !transform.any_value {transform.readonly},
                                         %arg3: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg1, "output 0" : !transform.any_value
    transform.debug.emit_remark_at %arg3, "output producer" : !transform.any_op
    transform.debug.emit_remark_at %arg2, "all output" : !transform.any_value
    transform.yield
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
    transform.foreach_match in %arg0 @match_init -> @print_init : (!transform.any_op) -> !transform.any_op
    transform.yield
  }


  func.func @payload(%f32: f32,
            // expected-remark @below {{output 0}}
            // expected-remark @below {{all output}}
            // expected-note @below {{value handle points to a block argument #1 in block #0 in region #0}}
            %tf32: tensor<?xf32>,
            // expected-remark @below {{all output}}
            // expected-note @below {{value handle points to a block argument #2 in block #0 in region #0}}
            %tf32_2: tensor<?xf32>)
            attributes { transform.target_tag = "start_here" }  {
    // expected-remark @below {{output 0}}
    // expected-remark @below {{output producer}}
    // expected-remark @below {{all output}}
    // expected-note @below {{value handle points to an op result #0}}
    %0 = linalg.fill ins(%f32: f32) outs(%tf32: tensor<?xf32>) -> tensor<?xf32>

    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%tf32: tensor<?xf32>) outs(%0, %tf32_2: tensor<?xf32>, tensor<?xf32>) {
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
      linalg.yield %arg0, %arg0 : f32, f32
    } -> (tensor<?xf32>, tensor<?xf32>)
    return
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @match_init_0_permutation(%arg0: !transform.any_op {transform.readonly})
      -> !transform.any_op {
    %0 = transform.match.structured failures(propagate) %arg0 : (!transform.any_op) -> !transform.any_op {
    ^bb0(%arg1: !transform.any_op):
      transform.match.structured.init %arg1[0] { permutation }: !transform.any_op
      transform.match.structured.yield %arg1 : !transform.any_op
    }
    transform.yield %0 : !transform.any_op
  }
  transform.named_sequence @match_init_1_permutation(%arg0: !transform.any_op {transform.readonly})
      -> !transform.any_op {
    %0 = transform.match.structured failures(propagate) %arg0 : (!transform.any_op) -> !transform.any_op {
    ^bb0(%arg1: !transform.any_op):
      transform.match.structured.init %arg1[1] { permutation }: !transform.any_op
      transform.match.structured.yield %arg1 : !transform.any_op
    }
    transform.yield %0 : !transform.any_op
  }
  transform.named_sequence @match_init_2_projected_permutation(%arg0: !transform.any_op {transform.readonly})
      -> !transform.any_op {
    %0 = transform.match.structured failures(propagate) %arg0 : (!transform.any_op) -> !transform.any_op {
    ^bb0(%arg1: !transform.any_op):
      transform.match.structured.init %arg1[2] { projected_permutation }: !transform.any_op
      transform.match.structured.yield %arg1 : !transform.any_op
    }
    transform.yield %0 : !transform.any_op
  }

  transform.named_sequence @print_init_0_permutation(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "matched output 0 permutation" : !transform.any_op
    transform.yield
  }
  transform.named_sequence @print_init_1_permutation(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "matched output 1 permutation" : !transform.any_op
    transform.yield
  }
  transform.named_sequence @print_init_2_projected_permutation(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "matched output 2 projected permutation" : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
    %0 = transform.foreach_match in %arg0 @match_init_0_permutation -> @print_init_0_permutation : (!transform.any_op) -> !transform.any_op
    %1 = transform.foreach_match in %0 @match_init_1_permutation -> @print_init_1_permutation : (!transform.any_op) -> !transform.any_op
    %2 = transform.foreach_match in %1 @match_init_2_projected_permutation -> @print_init_2_projected_permutation : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @payload(%f32: f32,
            %oned: tensor<?xf32>,
            %oned2: tensor<?xf32>,
            %twod: tensor<?x?xf32>)
            attributes { transform.target_tag = "start_here" }  {
    // expected-remark @below {{matched output 2 projected permutation}}
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0 + d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d1, d0)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%oned: tensor<?xf32>) outs(%oned, %oned2, %twod: tensor<?xf32>, tensor<?xf32>, tensor<?x?xf32>) {
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32):
      linalg.yield %arg0, %arg0, %arg0 : f32, f32, f32
    } -> (tensor<?xf32>, tensor<?xf32>, tensor<?x?xf32>)

    // expected-remark @below {{matched output 2 projected permutation}}
    // expected-remark @below {{matched output 1 permutation}}
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0 + d1)>,
                       affine_map<(d0, d1) -> (d1, d0)>,
                       affine_map<(d0, d1) -> (d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%oned: tensor<?xf32>) outs(%oned, %twod, %oned2: tensor<?xf32>, tensor<?x?xf32>, tensor<?xf32>) {
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32):
      linalg.yield %arg0, %arg0, %arg0 : f32, f32, f32
    } -> (tensor<?xf32>,  tensor<?x?xf32>, tensor<?xf32>)
    return
  }
}

// -----



module attributes { transform.with_named_sequence } {
  transform.named_sequence @match_num_io(%arg0: !transform.any_op {transform.readonly})
      -> (!transform.param<i64>, !transform.param<i64>, !transform.any_op) {
    %0:3 = transform.match.structured failures(propagate) %arg0
         : (!transform.any_op) -> (!transform.param<i64>, !transform.param<i64>, !transform.any_op) {
    ^bb0(%arg1: !transform.any_op):
      %1 = transform.match.structured.num_inputs %arg1 : (!transform.any_op) -> !transform.param<i64>
      %2 = transform.match.structured.num_inits %arg1 : (!transform.any_op) -> !transform.param<i64>
      transform.match.structured.yield %1, %2, %arg1 : !transform.param<i64>, !transform.param<i64>, !transform.any_op
    }
    transform.yield %0#0, %0#1, %0#2 : !transform.param<i64>, !transform.param<i64>, !transform.any_op
  }


  transform.named_sequence @print_num_io(
      %arg0: !transform.param<i64> {transform.readonly},
      %arg1: !transform.param<i64> {transform.readonly},
      %arg2: !transform.any_op {transform.readonly}) {
    transform.debug.emit_param_as_remark %arg0, "inputs" at %arg2 : !transform.param<i64>, !transform.any_op
    transform.debug.emit_param_as_remark %arg1, "outputs" at %arg2 : !transform.param<i64>, !transform.any_op
    transform.yield
  }


  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
    %0 = transform.foreach_match in %arg0 @match_num_io -> @print_num_io : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @payload(%f32: f32,
            %oned: tensor<?xf32>,
            %oned2: tensor<?xf32>,
            %twod: tensor<?x?xf32>)
            attributes { transform.target_tag = "start_here" }  {
    // expected-remark @below {{inputs 1}}
    // expected-remark @below {{outputs 3}}
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0 + d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d1, d0)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%oned: tensor<?xf32>) outs(%oned, %oned2, %twod: tensor<?xf32>, tensor<?xf32>, tensor<?x?xf32>) {
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32):
      linalg.yield %arg0, %arg0, %arg0 : f32, f32, f32
    } -> (tensor<?xf32>, tensor<?xf32>, tensor<?x?xf32>)

    // expected-remark @below {{inputs 2}}
    // expected-remark @below {{outputs 2}}
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d1, d0)>,
                       affine_map<(d0, d1) -> (d0 + d1)>,
                       affine_map<(d0, d1) -> (d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%oned, %twod: tensor<?xf32>, tensor<?x?xf32>) outs(%oned, %oned2: tensor<?xf32>, tensor<?xf32>) {
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32):
      linalg.yield %arg0, %arg0 : f32, f32
    } -> (tensor<?xf32>, tensor<?xf32>)
    return
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @match_rank(%arg0: !transform.any_op {transform.readonly})
      -> (!transform.param<i64>, !transform.any_op) {
    %0:2 = transform.match.structured failures(propagate) %arg0
         : (!transform.any_op) -> (!transform.param<i64>, !transform.any_op) {
    ^bb0(%arg1: !transform.any_op):
      %1 = transform.match.structured.rank %arg1 : (!transform.any_op) -> !transform.param<i64>
      transform.match.structured.yield %1, %arg1 : !transform.param<i64>, !transform.any_op
    }
    transform.yield %0#0, %0#1 : !transform.param<i64>, !transform.any_op
  }


  transform.named_sequence @print_rank(%arg0: !transform.param<i64> {transform.readonly},
                                       %arg2: !transform.any_op {transform.readonly}) {
    transform.debug.emit_param_as_remark %arg0, "rank" at %arg2 : !transform.param<i64>, !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
    %0 = transform.foreach_match in %arg0 @match_rank -> @print_rank : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @payload(%f32: f32,
            %twod: tensor<42x42xf32>)
            attributes { transform.target_tag = "start_here" } {
    %0 = tensor.empty() : tensor<42x42xf32>
    // expected-remark @below {{rank 2}}
    %1 = linalg.fill ins(%f32 : f32) outs(%0 : tensor<42x42xf32>) -> tensor<42x42xf32>
    // expected-remark @below {{rank 3}}
    linalg.matmul ins(%twod, %twod : tensor<42x42xf32>, tensor<42x42xf32>)
                  outs(%1 : tensor<42x42xf32>) -> tensor<42x42xf32>
    return
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @match_single_result(%arg0: !transform.any_op {transform.readonly})
      -> (!transform.any_op, !transform.any_op) {
    %0:2 = transform.match.structured failures(propagate) %arg0
         : (!transform.any_op) -> (!transform.any_op, !transform.any_op) {
    ^bb0(%arg1: !transform.any_op):
      %1 = transform.match.structured.result %arg1[0] { single } : (!transform.any_op) -> !transform.any_op
      transform.match.structured.yield %1, %arg1 : !transform.any_op, !transform.any_op
    }
    transform.yield %0#0, %0#1 : !transform.any_op, !transform.any_op
  }
  transform.named_sequence @match_result_value(%arg0: !transform.any_op {transform.readonly})
      -> (!transform.any_value, !transform.any_op) {
    %0:2 = transform.match.structured failures(propagate) %arg0
         : (!transform.any_op) -> (!transform.any_value, !transform.any_op) {
    ^bb0(%arg1: !transform.any_op):
      %1 = transform.match.structured.result %arg1[0] : (!transform.any_op) -> !transform.any_value
      transform.match.structured.yield %1, %arg1 : !transform.any_value, !transform.any_op
    }
    transform.yield %0#0, %0#1 : !transform.any_value, !transform.any_op
  }
  transform.named_sequence @match_any_result(%arg0: !transform.any_op {transform.readonly})
      -> (!transform.any_op) {
    %0 = transform.match.structured failures(propagate) %arg0
         : (!transform.any_op) -> !transform.any_op {
    ^bb0(%arg1: !transform.any_op):
      %1 = transform.match.structured.result %arg1[-1] { any } : (!transform.any_op) -> !transform.any_op
      transform.match.structured.yield %arg1 : !transform.any_op
    }
    transform.yield %0 : !transform.any_op
  }

  transform.named_sequence @print_single_result(%arg0: !transform.any_op {transform.readonly},
                                                %arg2: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg2, "matched single result" : !transform.any_op
    transform.debug.emit_remark_at %arg0, "single user" : !transform.any_op
    transform.yield
  }
  transform.named_sequence @print_result_value(%arg0: !transform.any_value {transform.readonly},
                                               %arg1: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg1, "matched result value" : !transform.any_op
    transform.debug.emit_remark_at %arg0, "op result" : !transform.any_value
    transform.yield
  }
  transform.named_sequence @print_any_result(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "matched any result" : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
    %0 = transform.foreach_match in %arg0 @match_single_result -> @print_single_result : (!transform.any_op) -> !transform.any_op
    %1 = transform.foreach_match in %0 @match_result_value -> @print_result_value : (!transform.any_op) -> !transform.any_op
    %2 = transform.foreach_match in %1 @match_any_result -> @print_any_result : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @payload(%f32: f32, %f322: f32, %f323: f32,
            %twod: tensor<42x42xf32>)
            attributes { transform.target_tag = "start_here" } {
    %0 = tensor.empty() : tensor<42x42xf32>

    // expected-remark @below {{matched result value}}
    // expected-remark @below {{op result}}
    // expected-note @below {{value handle points to an op result #0}}
    %1 = linalg.fill ins(%f32 : f32) outs(%0 : tensor<42x42xf32>) -> tensor<42x42xf32>
    // expected-remark @below {{matched result value}}
    // expected-remark @below {{op result}}
    // expected-note @below {{value handle points to an op result #0}}
    // expected-remark @below {{matched single result}}
    // expected-remark @below {{matched any result}}
    %2 = linalg.fill ins(%f322 : f32) outs(%0 : tensor<42x42xf32>) -> tensor<42x42xf32>
    // expected-remark @below {{matched result value}}
    // expected-remark @below {{op result}}
    // expected-note @below {{value handle points to an op result #0}}
    // expected-remark @below {{matched any result}}
    %3 = linalg.fill ins(%f323 : f32) outs(%0 : tensor<42x42xf32>) -> tensor<42x42xf32>

    // expected-remark @below {{matched result value}}
    // expected-remark @below {{op result}}
    // expected-note @below {{value handle points to an op result #0}}
    // expected-remark @below {{single user}}
    linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%2 : tensor<42x42xf32>) outs(%0 : tensor<42x42xf32>) -> tensor<42x42xf32>
    // expected-remark @below {{matched result value}}
    // expected-remark @below {{op result}}
    // expected-note @below {{value handle points to an op result #0}}
    linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%3 : tensor<42x42xf32>) outs(%0 : tensor<42x42xf32>) -> tensor<42x42xf32>
    // expected-remark @below {{matched result value}}
    // expected-remark @below {{op result}}
    // expected-note @below {{value handle points to an op result #0}}
    linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%3 : tensor<42x42xf32>) outs(%0 : tensor<42x42xf32>) -> tensor<42x42xf32>
    return
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @match_input_indexing_map(%arg0: !transform.any_op {transform.readonly})
      -> (!transform.affine_map, !transform.any_op) {
    %0 = transform.match.structured failures(propagate) %arg0
         : (!transform.any_op) -> !transform.affine_map {
    ^bb0(%arg1: !transform.any_op):
      %1 = transform.match.structured.input %arg1[0]  : (!transform.any_op) -> !transform.affine_map
      transform.match.structured.yield %1 : !transform.affine_map
    }
    transform.yield %0, %arg0 : !transform.affine_map, !transform.any_op
  }
  transform.named_sequence @match_init_indexing_map(%arg0: !transform.any_op {transform.readonly})
      -> (!transform.affine_map, !transform.any_op) {
    %0 = transform.match.structured failures(propagate) %arg0
         : (!transform.any_op) -> !transform.affine_map {
    ^bb0(%arg1: !transform.any_op):
      %1 = transform.match.structured.init %arg1[0]  : (!transform.any_op) -> !transform.affine_map
      transform.match.structured.yield %1 : !transform.affine_map
    }
    transform.yield %0, %arg0 : !transform.affine_map, !transform.any_op
  }

  transform.named_sequence @print_indexing_map_1(%arg0: !transform.affine_map {transform.readonly},
                                               %arg1: !transform.any_op {transform.readonly}) {
    transform.debug.emit_param_as_remark %arg0, "indexing map 1" at %arg1 : !transform.affine_map, !transform.any_op
    transform.yield
  }
  transform.named_sequence @print_indexing_map_2(%arg0: !transform.affine_map {transform.readonly},
                                               %arg1: !transform.any_op {transform.readonly}) {
    transform.debug.emit_param_as_remark %arg0, "indexing map 2" at %arg1 : !transform.affine_map, !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
    %3 = transform.foreach_match in %arg0 @match_input_indexing_map -> @print_indexing_map_1 : (!transform.any_op) -> !transform.any_op
    %4 = transform.foreach_match in %3 @match_init_indexing_map -> @print_indexing_map_2 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  func.func @payload(%lhs: tensor<32x32xf32>, %rhs: tensor<32x32xf32>)
            attributes { transform.target_tag = "start_here" } {
    %out = tensor.empty() : tensor<32x32xf32>
    %cst = arith.constant 1.0 : f32
    // expected-remark @below {{indexing map 1 affine_map<(d0, d1) -> ()>}}
    // expected-remark @below {{indexing map 2 affine_map<(d0, d1) -> (d0, d1)>}}
    %res = linalg.fill ins(%cst : f32) outs(%out : tensor<32x32xf32>) -> tensor<32x32xf32>
    // expected-remark @below {{indexing map 1 affine_map<(d0, d1, d2) -> (d0, d2)>}}
    // expected-remark @below {{indexing map 2 affine_map<(d0, d1, d2) -> (d0, d1)>}}
    linalg.matmul ins(%lhs, %rhs : tensor<32x32xf32>, tensor<32x32xf32>) outs(%res : tensor<32x32xf32>) -> tensor<32x32xf32>
    return
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @match_contraction(%arg0: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>) {
    %1:4 = transform.match.structured %arg0 : (!transform.any_op) -> (!transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>) {
    ^bb0(%struct: !transform.any_op):
      transform.match.structured.body %struct { contraction = ["arith.mulf", "arith.addf"] } : !transform.any_op
      %0:4 = transform.match.structured.classify_contraction_dims %struct
        : (!transform.any_op) -> (!transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
      transform.match.structured.yield %0#0, %0#1, %0#2, %0#3
        : !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>
    }
    transform.yield %arg0, %1#0, %1#1, %1#2, %1#3 : !transform.any_op, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>
  }

  transform.named_sequence @print_contraction(
      %op: !transform.any_op {transform.readonly},
      %batch: !transform.param<i64> {transform.readonly},
      %m: !transform.param<i64> {transform.readonly},
      %n: !transform.param<i64> {transform.readonly},
      %k: !transform.param<i64> {transform.readonly}) {
    transform.debug.emit_remark_at %op, "contraction" : !transform.any_op
    transform.debug.emit_param_as_remark %batch, "batch dims" at %op : !transform.param<i64>, !transform.any_op
    transform.debug.emit_param_as_remark %m, "m dims" at %op : !transform.param<i64>, !transform.any_op
    transform.debug.emit_param_as_remark %n, "n dims" at %op : !transform.param<i64>, !transform.any_op
    transform.debug.emit_param_as_remark %k, "k dims" at %op : !transform.param<i64>, !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
    %3 = transform.foreach_match in %arg0 @match_contraction -> @print_contraction : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

module attributes { transform.target_tag = "start_here" } {
  func.func @matmul_simple(%lhs: tensor<10x20xf32>, %rhs: tensor<20x15xf32>) -> tensor<10x15xf64> {
    %cst = arith.constant 0.0 : f64
    %empty = tensor.empty() : tensor<10x15xf64>
    %fill = linalg.fill ins(%cst : f64) outs(%empty : tensor<10x15xf64>) -> tensor<10x15xf64>
    // expected-remark @below {{contraction}}
    // expected-remark @below {{batch dims}}
    // expected-remark @below {{m dims 0}}
    // expected-remark @below {{n dims 1}}
    // expected-remark @below {{k dims 2}}
    %result = linalg.matmul ins(%lhs, %rhs: tensor<10x20xf32>, tensor<20x15xf32>) outs(%fill: tensor<10x15xf64>) -> tensor<10x15xf64>
    return %result : tensor<10x15xf64>
  }

  func.func @vecmat_simple(%lhs: tensor<20xf32>, %rhs: tensor<20x15xf32>) -> tensor<15xf64> {
    %cst = arith.constant 0.0 : f64
    %empty = tensor.empty() : tensor<15xf64>
    %fill = linalg.fill ins(%cst : f64) outs(%empty : tensor<15xf64>) -> tensor<15xf64>
    // expected-remark @below {{contraction}}
    // expected-remark @below {{batch dims}}
    // expected-remark @below {{m dims}}
    // expected-remark @below {{n dims 0}}
    // expected-remark @below {{k dims 1}}
    %result = linalg.vecmat ins(%lhs, %rhs: tensor<20xf32>, tensor<20x15xf32>) outs(%fill: tensor<15xf64>) -> tensor<15xf64>
    return %result : tensor<15xf64>
  }

  func.func @double_batch(%lhs: tensor<40x10x50x20xf32>, %rhs: tensor<40x20x50x15xf32>) -> tensor<40x10x50x15xf32> {
    %cst = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<40x10x50x15xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<40x10x50x15xf32>) -> tensor<40x10x50x15xf32>
    // expected-remark @below {{contraction}}
    // expected-remark @below {{batch dims 0 : i64, 2 : i64}}
    // expected-remark @below {{m dims 1}}
    // expected-remark @below {{n dims 3}}
    // expected-remark @below {{k dims 4}}
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>,
                      affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2, d3)>,
                      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
    } ins(%lhs, %rhs : tensor<40x10x50x20xf32>, tensor<40x20x50x15xf32>)
      outs(%fill : tensor<40x10x50x15xf32>) {
    ^bb(%arg0: f32, %arg1: f32, %arg2: f32):
      %0 = arith.mulf %arg0, %arg1 : f32
      %1 = arith.addf %arg2, %0 : f32
      linalg.yield %1 : f32
    } -> tensor<40x10x50x15xf32>
    return %result : tensor<40x10x50x15xf32>
  }

  func.func @generic_min(%arg0: tensor<1x7x4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<1x1x4xf32>) {
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1 * 2 + d3 * 2, d2)>, 
      affine_map<(d0, d1, d2, d3) -> (d3)>, 
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], 
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]} 
      ins(%arg0, %arg1 : tensor<1x7x4xf32>, tensor<4xf32>) 
      outs(%arg2 : tensor<1x1x4xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %5 = arith.minimumf %out, %in : f32
      linalg.yield %5 : f32
    } -> tensor<1x1x4xf32>
    return
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @match_convolution(%arg0: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>) {
    %1:8 = transform.match.structured %arg0 : (!transform.any_op) -> (!transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>) {
    ^bb0(%struct: !transform.any_op):
      transform.match.structured.body %struct { contraction = ["arith.mulf", "arith.addf"] } : !transform.any_op
      %0:8 = transform.match.structured.classify_convolution_dims %struct
        : (!transform.any_op) -> (!transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
      transform.match.structured.yield %0#0, %0#1, %0#2, %0#3, %0#4, %0#5, %0#6, %0#7
        : !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>
    }
    transform.yield %arg0, %1#0, %1#1, %1#2, %1#3, %1#4, %1#5, %1#6, %1#7 : !transform.any_op, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>
  }

  transform.named_sequence @print_convolution(
      %op: !transform.any_op {transform.readonly},
      %batch: !transform.param<i64> {transform.readonly},
      %oi: !transform.param<i64> {transform.readonly},
      %oc: !transform.param<i64> {transform.readonly},
      %fl: !transform.param<i64> {transform.readonly},
      %ic: !transform.param<i64> {transform.readonly},
      %depth: !transform.param<i64> {transform.readonly},
      %strides: !transform.param<i64> {transform.readonly},
      %dilations: !transform.param<i64> {transform.readonly}) {
    transform.debug.emit_remark_at %op, "convolution" : !transform.any_op
    transform.debug.emit_param_as_remark %batch, "batch dims" at %op : !transform.param<i64>, !transform.any_op
    transform.debug.emit_param_as_remark %oi, "output image dims" at %op : !transform.param<i64>, !transform.any_op
    transform.debug.emit_param_as_remark %oc, "output channel dims" at %op : !transform.param<i64>, !transform.any_op
    transform.debug.emit_param_as_remark %fl, "filter loop dims" at %op : !transform.param<i64>, !transform.any_op
    transform.debug.emit_param_as_remark %ic, "input channel dims" at %op : !transform.param<i64>, !transform.any_op
    transform.debug.emit_param_as_remark %depth, "depth dims" at %op : !transform.param<i64>, !transform.any_op
    transform.debug.emit_param_as_remark %strides, "strides" at %op : !transform.param<i64>, !transform.any_op
    transform.debug.emit_param_as_remark %dilations, "dilations" at %op : !transform.param<i64>, !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
    %3 = transform.foreach_match in %arg0 @match_convolution -> @print_convolution : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

module attributes { transform.target_tag = "start_here" } {
  func.func @convolution_simple(%input: tensor<10x20x30xf32>, %filter: tensor<3x30x15xf32>) -> tensor<10x18x15xf64> {
    %cst = arith.constant 0.0 : f64
    %empty = tensor.empty() : tensor<10x18x15xf64>
    %fill = linalg.fill ins(%cst : f64) outs(%empty : tensor<10x18x15xf64>) -> tensor<10x18x15xf64>
    // expected-remark @below {{convolution}}
    // expected-remark @below {{batch dims 0}}
    // expected-remark @below {{output image dims 1}}
    // expected-remark @below {{output channel dims 2}}
    // expected-remark @below {{filter loop dims 3}}
    // expected-remark @below {{input channel dims 4}}
    // expected-remark @below {{depth dims}}
    // expected-remark @below {{strides 1}}
    // expected-remark @below {{dilations 1}}
    %result = linalg.conv_1d_nwc_wcf {dilations = dense<1> : tensor<1xi64>,
                                      strides = dense<1> : tensor<1xi64>}
       ins(%input, %filter: tensor<10x20x30xf32>, tensor<3x30x15xf32>) outs(%fill: tensor<10x18x15xf64>) -> tensor<10x18x15xf64>
    return %result : tensor<10x18x15xf64>
  }

  func.func @convolution_depthwise(%input: tensor<1x10x196x48xf32>, %filter: tensor<1x4x48xf32>) -> tensor<1x10x191x48xf32> {
    %cst = arith.constant 0.0 : f32 
    %empty = tensor.empty() : tensor<1x10x191x48xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x10x191x48xf32>) -> tensor<1x10x191x48xf32>
    // expected-remark @below {{convolution}}
    // expected-remark @below {{batch dims 0}}
    // expected-remark @below {{output image dims 1 : i64, 2 : i64}}
    // expected-remark @below {{output channel dims}}
    // expected-remark @below {{filter loop dims 4 : i64, 5 : i64}}
    // expected-remark @below {{input channel dims}}
    // expected-remark @below {{depth dims 3}}
    // expected-remark @below {{strides 1 : i64, 1 : i64}}
    // expected-remark @below {{dilations 1 : i64, 1 : i64}}
    %result = linalg.depthwise_conv_2d_nhwc_hwc {
      dilations = dense<1> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x10x196x48xf32>, tensor<1x4x48xf32>)
      outs(%fill : tensor<1x10x191x48xf32>) -> tensor<1x10x191x48xf32>

    return %result : tensor<1x10x191x48xf32>
  }

  func.func @convolution_multi_channel(%input: tensor<2x34x68x16xf32>, %filter: tensor<8x2x3x5x16x16xf32>) -> tensor<8x32x32x16xf32> {
    %cst = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<8x32x32x16xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<8x32x32x16xf32>) -> tensor<8x32x32x16xf32>
    // expected-remark @below {{convolution}}
    // expected-remark @below {{batch dims}}
    // expected-remark @below {{output image dims 1 : i64, 2 : i64}}
    // expected-remark @below {{output channel dims 0 : i64, 3 : i64}}
    // expected-remark @below {{filter loop dims 5 : i64, 6 : i64}}
    // expected-remark @below {{input channel dims 4 : i64, 7 : i64}}
    // expected-remark @below {{depth dims}}
    // expected-remark @below {{strides 1 : i64, 2 : i64}}
    // expected-remark @below {{dilations 1 : i64, 1 : i64}}
    %result = linalg.generic {
        indexing_maps = [
            affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d1 + d5, 2 * d2 + d6, d7)>,
            affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d4, d5, d6, d7, d3)>,
            affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]}
        ins(%input, %filter : tensor<2x34x68x16xf32>, tensor<8x2x3x5x16x16xf32>) outs(%fill : tensor<8x32x32x16xf32>) {
          ^bb0(%in: f32, %in_0: f32, %out: f32):
            %mul = arith.mulf %in, %in_0 : f32
            %add = arith.addf %mul, %out : f32
            linalg.yield %add : f32
          } -> tensor<8x32x32x16xf32>
    return %result : tensor<8x32x32x16xf32>
  }
}
