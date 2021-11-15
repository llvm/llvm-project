// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: @test_roundtrip_parameter_parsers
// CHECK: !test.type_with_format<111, three = #test<"attr_ugly begin 5 : index end">, two = "foo">
// CHECK: !test.type_with_format<2147, three = "hi", two = "hi">
func private @test_roundtrip_parameter_parsers(!test.type_with_format<111, three = #test<"attr_ugly begin 5 : index end">, two = "foo">) -> !test.type_with_format<2147, two = "hi", three = "hi">
attributes {
  // CHECK: #test.attr_with_format<3 : two = "hello", four = [1, 2, 3] : 42 : i64>
  attr0 = #test.attr_with_format<3 : two = "hello", four = [1, 2, 3] : 42 : i64>,
  // CHECK: #test.attr_with_format<5 : two = "a_string", four = [4, 5, 6, 7, 8] : 8 : i8>
  attr1 = #test.attr_with_format<5 : two = "a_string", four = [4, 5, 6, 7, 8] : 8 : i8>,
  // CHECK: #test<"attr_ugly begin 5 : index end">
  attr2 = #test<"attr_ugly begin 5 : index end">,
  // CHECK: #test.attr_params<42, 24>
  attr3 = #test.attr_params<42, 24>,
  // CHECK: #test.attr_with_type<i32, vector<4xi32>>
  attr4 = #test.attr_with_type<i32, vector<4xi32>>
}

// CHECK-LABEL: @test_roundtrip_default_parsers_struct
// CHECK: !test.no_parser<255, [1, 2, 3, 4, 5], "foobar", 4>
// CHECK: !test.struct_capture_all<v0 = 0, v1 = 1, v2 = 2, v3 = 3>
func private @test_roundtrip_default_parsers_struct(!test.no_parser<255, [1, 2, 3, 4, 5], "foobar", 4>) -> !test.struct_capture_all<v3 = 3, v1 = 1, v2 = 2, v0 = 0>
