// RUN: mlir-opt %s --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics

func.func @bar() {
  // expected-remark @below {{matched op name}}
  // expected-remark @below {{matched attr name}}
  %0 = arith.constant {my_attr} 0: i32
  // expected-remark @below {{matched op name}}
  %1 = arith.constant 1 : i32
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %match_name = transform.structured.match ops{["arith.constant"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %match_name, "matched op name" : !transform.any_op
    transform.test_consume_operand %match_name : !transform.any_op

    %match_attr = transform.structured.match ops{["arith.constant"]} attributes{my_attr} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %match_attr, "matched attr name" : !transform.any_op
    transform.test_consume_operand %match_attr : !transform.any_op
    transform.yield
  }
}

// -----

func.func @by_type() {
  %0 = arith.constant 0: i32
  // expected-remark @below {{matched op name}}
  %1 = arith.constant 1.0 : f32
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %match_name = transform.structured.match
      ops{["arith.constant"]} filter_result_type = f32 in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %match_name, "matched op name" : !transform.any_op
    transform.test_consume_operand %match_name : !transform.any_op
    transform.yield
  }
}

// -----

func.func @by_operand_type() {
  %c2 = arith.constant 2.0: f32
  %v = arith.constant 8: i32
  %r1 = math.fpowi %c2, %v : f32, i32
  // expected-remark @below {{matched op name}}
  %r2 = arith.addf %c2, %c2 : f32
  // expected-remark @below {{matched op name}}
  %r3 = arith.fptoui %r2 : f32 to i32
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %match_name1 = transform.structured.match
      ops{["arith.fptoui"]} filter_operand_types = [f32] in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %match_name1, "matched op name" : !transform.any_op
    transform.test_consume_operand %match_name1 : !transform.any_op

    %match_name2 = transform.structured.match
      ops{["arith.addf"]} filter_operand_types = [f32] in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %match_name2, "matched op name" : !transform.any_op
    transform.test_consume_operand %match_name2 : !transform.any_op

    %no_match_name1 = transform.structured.match
      ops{["arith.fptoui"]} filter_operand_types = [i32] in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %no_match_name1, "should not match" : !transform.any_op
    transform.test_consume_operand %no_match_name1 : !transform.any_op

    %no_match_name2 = transform.structured.match
      ops{["math.fpowi"]} filter_operand_types = [f32] in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %no_match_name2, "should not match" : !transform.any_op
    transform.test_consume_operand %no_match_name2 : !transform.any_op
    transform.yield
  }
}

// -----

func.func @foo(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>) {
  %c0 = arith.constant 0.0 : f32
  // expected-remark @below {{tileable}}
  %r = linalg.fill ins(%c0 : f32) outs(%c : tensor<4x4xf32>) -> tensor<4x4xf32>
  // expected-remark @below {{tileable}}
  linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%r : tensor<4x4xf32>) -> tensor<4x4xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match interface{TilingInterface} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %matched, "tileable" : !transform.any_op
    transform.yield
  }
}

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
func.func @match_complex_attribute(%arg0: tensor<12x128x32xf32>)
    -> tensor<128x12x32xf32> {
  %0 = tensor.empty() : tensor<128x12x32xf32>
  // expected-remark @below {{matched complex attr}}
  %1 = linalg.generic {indexing_maps = [#map0, #map1],
                       iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg0 : tensor<12x128x32xf32>)
    outs(%0 : tensor<128x12x32xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    linalg.yield %arg1 : f32
  } -> tensor<128x12x32xf32>
  return %1 : tensor<128x12x32xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %match_attr = transform.structured.match
        ops{["linalg.generic"]}
        attributes{iterator_types = [
          #linalg.iterator_type<parallel>,
          #linalg.iterator_type<parallel>,
          #linalg.iterator_type<parallel>]}
        in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %match_attr, "matched complex attr" : !transform.any_op
    transform.test_consume_operand %match_attr : !transform.any_op

    %no_match = transform.structured.match
        attributes{iterator_types = [
          #linalg.iterator_type<parallel>,
          #linalg.iterator_type<parallel>,
          #linalg.iterator_type<reduction>]}
        in %arg1 : (!transform.any_op) -> !transform.any_op
    %p = transform.num_associations %no_match : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below {{0}}
    transform.debug.emit_param_as_remark %p : !transform.param<i64>
    transform.yield
  }
}

// -----

func.func private @callee()

func.func @foo(%lb: index, %ub: index, %step: index) {
  // expected-remark @below {{loop-like}}
  scf.for %i = %lb to %ub step %step {
    func.call @callee() : () -> ()
    scf.yield
  }
  // expected-remark @below {{loop-like}}
  scf.parallel (%i) = (%lb) to (%ub) step (%step) {
    func.call @callee() : () -> ()
    scf.reduce
  }
  // expected-remark @below {{loop-like}}
  scf.forall (%i) in (%ub) {
    func.call @callee() : () -> ()
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match interface{LoopLikeInterface} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %matched, "loop-like" : !transform.any_op
    transform.yield
  }
}
