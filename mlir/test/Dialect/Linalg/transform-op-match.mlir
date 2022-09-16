// RUN: mlir-opt %s --test-transform-dialect-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics

func.func @bar() {
  // expected-remark @below {{matched op name}}
  // expected-remark @below {{matched attr name}}
  %0 = arith.constant {my_attr} 0: i32
  // expected-remark @below {{matched op name}}
  %1 = arith.constant 1 : i32
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %match_name = transform.structured.match ops{["arith.constant"]} in %arg1
    transform.test_print_remark_at_operand %match_name, "matched op name"
    transform.test_consume_operand %match_name

    %match_attr = transform.structured.match ops{["arith.constant"]} attributes{my_attr} in %arg1
    transform.test_print_remark_at_operand %match_attr, "matched attr name"
    transform.test_consume_operand %match_attr
  }
}

// -----

func.func @by_type() {
  %0 = arith.constant 0: i32
  // expected-remark @below {{matched op name}}
  %1 = arith.constant 1.0 : f32
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %match_name = transform.structured.match
      ops{["arith.constant"]} filter_result_type = f32 in %arg1
    transform.test_print_remark_at_operand %match_name, "matched op name"
    transform.test_consume_operand %match_name
  }
}

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
func.func @match_complex_attribute(%arg0: tensor<12x128x32xf32>)
    -> tensor<128x12x32xf32> {
  %0 = linalg.init_tensor [128, 12, 32] : tensor<128x12x32xf32>
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

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %match_attr = transform.structured.match
        ops{["linalg.generic"]}
        attributes{iterator_types = ["parallel", "parallel", "parallel"]}
        in %arg1
    transform.test_print_remark_at_operand %match_attr, "matched complex attr"
    transform.test_consume_operand %match_attr

    %no_match = transform.structured.match
        attributes{iterator_types = ["parallel", "parallel", "reduction"]}
        in %arg1
  // expected-remark @below {{0}}
    transform.test_print_number_of_associated_payload_ir_ops %no_match
  }
}
