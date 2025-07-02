// RUN: mlir-opt %s | FileCheck %s --check-prefix=CHECK-ROUNDTRIP
// RUN: mlir-opt %s -mlir-print-op-generic | FileCheck %s --check-prefix=CHECK-GENERIC-SYNTAX

/// This file tetss that "custom_dense_array" (which is a DenseArrayAttribute
/// stored within the attr-dict) is parsed and printed with the "pretty" array
/// syntax (i.e. `[1, 2, 3, 4]`), rather than with the generic dense array
/// syntax (`array<i64: 1, 2, 3, 4>`).
///
/// This is done by injecting custom parsing and printing callbacks into
/// parseOptionalAttrDict() and printOptionalAttrDict().

func.func @custom_attr_dict_syntax() {
  // CHECK-ROUNDTRIP: test.custom_attr_parse_and_print_in_attr_dict {custom_dense_array = [1, 2, 3, 4]}
  // CHECK-GENERIC-SYNTAX: "test.custom_attr_parse_and_print_in_attr_dict"() <{custom_dense_array = array<i64: 1, 2, 3, 4>}> : () -> ()
  test.custom_attr_parse_and_print_in_attr_dict {custom_dense_array = [1, 2, 3, 4]}

  // CHECK-ROUNDTRIP: test.custom_attr_parse_and_print_in_attr_dict {another_attr = "foo", custom_dense_array = [1, 2, 3, 4]}
  // CHECK-GENERIC-SYNTAX: "test.custom_attr_parse_and_print_in_attr_dict"() <{custom_dense_array = array<i64: 1, 2, 3, 4>}> {another_attr = "foo"} : () -> ()
  test.custom_attr_parse_and_print_in_attr_dict {another_attr = "foo", custom_dense_array = [1, 2, 3, 4]}

  // CHECK-ROUNDTRIP: test.custom_attr_parse_and_print_in_attr_dict {custom_dense_array = [1, 2, 3, 4], default_array = [1, 2, 3, 4]}
  // CHECK-GENERIC-SYNTAX: "test.custom_attr_parse_and_print_in_attr_dict"() <{custom_dense_array = array<i64: 1, 2, 3, 4>}> {default_array = [1, 2, 3, 4]} : () -> ()
  test.custom_attr_parse_and_print_in_attr_dict {custom_dense_array = [1, 2, 3, 4], default_array = [1, 2, 3, 4]}

  // CHECK-ROUND-TRIP: test.custom_attr_parse_and_print_in_attr_dict {default_dense_array = array<i64: 1, 2, 3, 4>, custom_dense_array = [1, 2, 3, 4]}
  // CHECK-GENERIC-SYNTAX: "test.custom_attr_parse_and_print_in_attr_dict"() <{custom_dense_array = array<i64: 1, 2, 3, 4>}> {default_dense_array = array<i64: 1, 2, 3, 4>} : () -> ()
  test.custom_attr_parse_and_print_in_attr_dict {default_dense_array = array<i64: 1, 2, 3, 4>, custom_dense_array = [1, 2, 3, 4]}

  return
}
