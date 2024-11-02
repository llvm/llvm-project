// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: @test_roundtrip_parameter_parsers
// CHECK: !test.type_with_format<111, three = #test<"attr_ugly begin 5 : index end">, two = "foo">
// CHECK: !test.type_with_format<2147, three = "hi", two = "hi">
func.func private @test_roundtrip_parameter_parsers(!test.type_with_format<111, three = #test<"attr_ugly begin 5 : index end">, two = "foo">) -> !test.type_with_format<2147, two = "hi", three = "hi">
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
  attr4 = #test.attr_with_type<i32, vector<4xi32>>,
  // CHECK: #test.attr_self_type_format<5> : i32
  attr5 = #test.attr_self_type_format<5> : i32
}

// CHECK-LABEL: @test_roundtrip_default_parsers_struct
// CHECK: !test.no_parser<255, [1, 2, 3, 4, 5], "foobar", 4>
// CHECK: !test.struct_capture_all<v0 = 0, v1 = 1, v2 = 2, v3 = 3>
// CHECK: !test.optional_param<, 6>
// CHECK: !test.optional_param<5, 6>
// CHECK: !test.optional_params<"a">
// CHECK: !test.optional_params<5, "a">
// CHECK: !test.optional_struct<b = "a">
// CHECK: !test.optional_struct<a = 5, b = "a">
// CHECK: !test.optional_params_after<"a">
// CHECK: !test.optional_params_after<"a", 5>
// CHECK: !test.all_optional_params<>
// CHECK: !test.all_optional_params<5>
// CHECK: !test.all_optional_params<5, 6>
// CHECK: !test.all_optional_struct<>
// CHECK: !test.all_optional_struct<b = 5>
// CHECK: !test.all_optional_struct<a = 5, b = 10>
// CHECK: !test.optional_group<(5) 6>
// CHECK: !test.optional_group<x 6>
// CHECK: !test.optional_group_params<x>
// CHECK: !test.optional_group_params<(5)>
// CHECK: !test.optional_group_params<(5, 6)>
// CHECK: !test.optional_group_struct<x>
// CHECK: !test.optional_group_struct<(b = 5)>
// CHECK: !test.optional_group_struct<(a = 10, b = 5)>
// CHECK: !test.spaces< 5
// CHECK-NEXT: ()() 6>
// CHECK: !test.ap_float<5.000000e+00>
// CHECK: !test.ap_float<>
// CHECK: !test.default_valued_type<(i64)>
// CHECK: !test.default_valued_type<>
// CHECK: !test.custom_type<-5>
// CHECK: !test.custom_type<2 0 1 5>
// CHECK: !test.custom_type_string<"foo" foo>
// CHECK: !test.custom_type_string<"bar" bar>

func.func private @test_roundtrip_default_parsers_struct(
  !test.no_parser<255, [1, 2, 3, 4, 5], "foobar", 4>
) -> (
  !test.struct_capture_all<v3 = 3, v1 = 1, v2 = 2, v0 = 0>,
  !test.optional_param<, 6>,
  !test.optional_param<5, 6>,
  !test.optional_params<"a">,
  !test.optional_params<5, "a">,
  !test.optional_struct<b = "a">,
  !test.optional_struct<b = "a", a = 5>,
  !test.optional_params_after<"a">,
  !test.optional_params_after<"a", 5>,
  !test.all_optional_params<>,
  !test.all_optional_params<5>,
  !test.all_optional_params<5, 6>,
  !test.all_optional_struct<>,
  !test.all_optional_struct<b = 5>,
  !test.all_optional_struct<b = 10, a = 5>,
  !test.optional_group<(5) 6>,
  !test.optional_group<x 6>,
  !test.optional_group_params<x>,
  !test.optional_group_params<(5)>,
  !test.optional_group_params<(5, 6)>,
  !test.optional_group_struct<x>,
  !test.optional_group_struct<(b = 5)>,
  !test.optional_group_struct<(b = 5, a = 10)>,
  !test.spaces<5 ()() 6>,
  !test.ap_float<5.0>,
  !test.ap_float<>,
  !test.default_valued_type<(i64)>,
  !test.default_valued_type<>,
  !test.custom_type<-5>,
  !test.custom_type<2 9 9 5>,
  !test.custom_type_string<"foo" foo>,
  !test.custom_type_string<"bar" bar>
)
