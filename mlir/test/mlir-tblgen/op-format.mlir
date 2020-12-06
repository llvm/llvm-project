// RUN: mlir-opt -allow-unregistered-dialect %s | mlir-opt -allow-unregistered-dialect -verify-diagnostics | FileCheck %s

// CHECK: %[[I64:.*]] =
%i64 = "foo.op"() : () -> (i64)
// CHECK: %[[I32:.*]] =
%i32 = "foo.op"() : () -> (i32)
// CHECK: %[[MEMREF:.*]] =
%memref = "foo.op"() : () -> (memref<1xf64>)

// CHECK: test.format_literal_op keyword_$. -> :, = <> () []( ) ? + * {foo.some_attr}
test.format_literal_op keyword_$. -> :, = <> () []( ) ? + * {foo.some_attr}

// CHECK: test.format_attr_op 10
// CHECK-NOT: {attr
test.format_attr_op 10

// CHECK: test.format_opt_attr_op_a(10)
// CHECK-NOT: {opt_attr
test.format_opt_attr_op_a(10)
test.format_opt_attr_op_a

// CHECK: test.format_opt_attr_op_b 10
// CHECK-NOT: {opt_attr
test.format_opt_attr_op_b 10
test.format_opt_attr_op_b

// CHECK: test.format_symbol_name_attr_op @name
// CHECK-NOT: {attr
test.format_symbol_name_attr_op @name

// CHECK: test.format_symbol_name_attr_op @opt_name
// CHECK-NOT: {attr
test.format_symbol_name_attr_op @opt_name
test.format_opt_symbol_name_attr_op

// CHECK: test.format_attr_dict_w_keyword attributes {attr = 10 : i64}
test.format_attr_dict_w_keyword attributes {attr = 10 : i64}

// CHECK: test.format_attr_dict_w_keyword attributes {attr = 10 : i64, opt_attr = 10 : i64}
test.format_attr_dict_w_keyword attributes {attr = 10 : i64, opt_attr = 10 : i64}

// CHECK: test.format_buildable_type_op %[[I64]]
%ignored = test.format_buildable_type_op %i64

//===----------------------------------------------------------------------===//
// Format regions
//===----------------------------------------------------------------------===//

// CHECK: test.format_region_a_op {
// CHECK-NEXT: test.return
test.format_region_a_op {
  "test.return"() : () -> ()
}

// CHECK: test.format_region_b_op {
// CHECK-NEXT: test.return
test.format_region_b_op {
  "test.return"() : () -> ()
}

// CHECK: test.format_region_c_op region {
// CHECK-NEXT: test.return
test.format_region_c_op region {
  "test.return"() : () -> ()
}
// CHECK: test.format_region_c_op
// CHECK-NOT: region {
test.format_region_c_op

// CHECK: test.format_variadic_region_a_op {
// CHECK-NEXT: test.return
// CHECK-NEXT: }, {
// CHECK-NEXT: test.return
// CHECK-NEXT: }
test.format_variadic_region_a_op {
  "test.return"() : () -> ()
}, {
  "test.return"() : () -> ()
}
// CHECK: test.format_variadic_region_b_op {
// CHECK-NEXT: test.return
// CHECK-NEXT: }, {
// CHECK-NEXT: test.return
// CHECK-NEXT: } found_regions
test.format_variadic_region_b_op {
  "test.return"() : () -> ()
}, {
  "test.return"() : () -> ()
} found_regions
// CHECK: test.format_variadic_region_b_op
// CHECK-NOT: {
// CHECK-NOT: found_regions
test.format_variadic_region_b_op

// CHECK: test.format_implicit_terminator_region_a_op {
// CHECK-NEXT: }
test.format_implicit_terminator_region_a_op {
  "test.return"() : () -> ()
}
// CHECK: test.format_implicit_terminator_region_a_op {
// CHECK-NEXT: test.return"() {foo.attr
test.format_implicit_terminator_region_a_op {
  "test.return"() {foo.attr} : () -> ()
}
// CHECK: test.format_implicit_terminator_region_a_op {
// CHECK-NEXT: test.return"(%[[I64]]) : (i64)
test.format_implicit_terminator_region_a_op {
  "test.return"(%i64) : (i64) -> ()
}

//===----------------------------------------------------------------------===//
// Format results
//===----------------------------------------------------------------------===//

// CHECK: test.format_result_a_op memref<1xf64>
%ignored_a:2 = test.format_result_a_op memref<1xf64>

// CHECK: test.format_result_b_op i64, memref<1xf64>
%ignored_b:2 = test.format_result_b_op i64, memref<1xf64>

// CHECK: test.format_result_c_op (i64) -> memref<1xf64>
%ignored_c:2 = test.format_result_c_op (i64) -> memref<1xf64>

// CHECK: test.format_variadic_result : i64, i64, i64
%ignored_v:3 = test.format_variadic_result : i64, i64, i64

// CHECK: test.format_multiple_variadic_results : (i64, i64, i64), (i32, i32)
%ignored_mv:5 = test.format_multiple_variadic_results : (i64, i64, i64), (i32, i32)

//===----------------------------------------------------------------------===//
// Format operands
//===----------------------------------------------------------------------===//

// CHECK: test.format_operand_a_op %[[I64]], %[[MEMREF]] : i64, memref<1xf64>
test.format_operand_a_op %i64, %memref : i64, memref<1xf64>

// CHECK: test.format_operand_b_op %[[I64]], %[[MEMREF]] : memref<1xf64>
test.format_operand_b_op %i64, %memref : memref<1xf64>

// CHECK: test.format_operand_c_op %[[I64]], %[[MEMREF]] : i64, memref<1xf64>
test.format_operand_c_op %i64, %memref : i64, memref<1xf64>

// CHECK: test.format_operand_d_op %[[I64]], %[[MEMREF]] : memref<1xf64>
test.format_operand_d_op %i64, %memref : memref<1xf64>

// CHECK: test.format_operand_e_op %[[I64]], %[[MEMREF]] : i64, memref<1xf64>
test.format_operand_e_op %i64, %memref : i64, memref<1xf64>

// CHECK: test.format_variadic_operand %[[I64]], %[[I64]], %[[I64]] : i64, i64, i64
test.format_variadic_operand %i64, %i64, %i64 : i64, i64, i64

// CHECK: test.format_multiple_variadic_operands (%[[I64]], %[[I64]], %[[I64]]), (%[[I64]], %[[I32]] : i64, i32)
test.format_multiple_variadic_operands (%i64, %i64, %i64), (%i64, %i32 : i64, i32)

//===----------------------------------------------------------------------===//
// Format successors
//===----------------------------------------------------------------------===//

"foo.successor_test_region"() ( {
  ^bb0:
    // CHECK: test.format_successor_a_op ^bb1 {attr}
    test.format_successor_a_op ^bb1 {attr}

  ^bb1:
    // CHECK: test.format_successor_a_op ^bb1, ^bb2 {attr}
    test.format_successor_a_op ^bb1, ^bb2 {attr}

  ^bb2:
    // CHECK: test.format_successor_a_op {attr}
    test.format_successor_a_op {attr}

}) { arg_names = ["i", "j", "k"] } : () -> ()

//===----------------------------------------------------------------------===//
// Format optional attributes
//===----------------------------------------------------------------------===//

// CHECK: test.format_optional_unit_attribute is_optional
test.format_optional_unit_attribute is_optional

// CHECK: test.format_optional_unit_attribute
// CHECK-NOT: is_optional
test.format_optional_unit_attribute

// CHECK: test.format_optional_unit_attribute_no_elide unit
test.format_optional_unit_attribute_no_elide unit

// CHECK: test.format_optional_enum_attr "case5"
test.format_optional_enum_attr "case5"

// CHECK: test.format_optional_enum_attr
// CHECK-NOT: "case5"
test.format_optional_enum_attr

//===----------------------------------------------------------------------===//
// Format optional operands and results
//===----------------------------------------------------------------------===//

// CHECK: test.format_optional_operand_result_a_op(%[[I64]] : i64) : i64
test.format_optional_operand_result_a_op(%i64 : i64) : i64

// CHECK: test.format_optional_operand_result_a_op( : ) : i64
test.format_optional_operand_result_a_op( : ) : i64

// CHECK: test.format_optional_operand_result_a_op(%[[I64]] : i64) :
// CHECK-NOT: i64
test.format_optional_operand_result_a_op(%i64 : i64) :

// CHECK: test.format_optional_operand_result_a_op(%[[I64]] : i64) : [%[[I64]], %[[I64]]]
test.format_optional_operand_result_a_op(%i64 : i64) : [%i64, %i64]

// CHECK: test.format_optional_operand_result_b_op(%[[I64]] : i64) : i64
test.format_optional_operand_result_b_op(%i64 : i64) : i64

// CHECK: test.format_optional_operand_result_b_op : i64
test.format_optional_operand_result_b_op( : ) : i64

// CHECK: test.format_optional_operand_result_b_op : i64
test.format_optional_operand_result_b_op : i64

//===----------------------------------------------------------------------===//
// Format custom directives
//===----------------------------------------------------------------------===//

// CHECK: test.format_custom_directive_operands %[[I64]], %[[I64]] -> (%[[I64]])
test.format_custom_directive_operands %i64, %i64 -> (%i64)

// CHECK: test.format_custom_directive_operands %[[I64]] -> (%[[I64]])
test.format_custom_directive_operands %i64 -> (%i64)

// CHECK: test.format_custom_directive_operands_and_types %[[I64]], %[[I64]] -> (%[[I64]]) : i64, i64 -> (i64)
test.format_custom_directive_operands_and_types %i64, %i64 -> (%i64) : i64, i64 -> (i64)

// CHECK: test.format_custom_directive_operands_and_types %[[I64]] -> (%[[I64]]) : i64 -> (i64)
test.format_custom_directive_operands_and_types %i64 -> (%i64) : i64 -> (i64)

// CHECK: test.format_custom_directive_attributes 54 : i64
test.format_custom_directive_attributes 54 : i64

// CHECK: test.format_custom_directive_attributes 54 : i64, 46 : i64
test.format_custom_directive_attributes 54 : i64, 46 : i64

// CHECK: test.format_custom_directive_regions {
// CHECK-NEXT: test.return
// CHECK-NEXT: }
test.format_custom_directive_regions {
  "test.return"() : () -> ()
}

// CHECK: test.format_custom_directive_regions {
// CHECK-NEXT: test.return
// CHECK-NEXT: }, {
// CHECK-NEXT: test.return
// CHECK-NEXT: }
test.format_custom_directive_regions {
  "test.return"() : () -> ()
}, {
  "test.return"() : () -> ()
}

// CHECK: test.format_custom_directive_results : i64, i64 -> (i64)
test.format_custom_directive_results : i64, i64 -> (i64)

// CHECK: test.format_custom_directive_results : i64 -> (i64)
test.format_custom_directive_results : i64 -> (i64)

// CHECK: test.format_custom_directive_results_with_type_refs : i64, i64 -> (i64) type_refs_capture : i64, i64 -> (i64)
test.format_custom_directive_results_with_type_refs : i64, i64 -> (i64) type_refs_capture : i64, i64 -> (i64)

// CHECK: test.format_custom_directive_results_with_type_refs : i64 -> (i64) type_refs_capture : i64 -> (i64)
test.format_custom_directive_results_with_type_refs : i64 -> (i64) type_refs_capture : i64 -> (i64)

func @foo() {
  // CHECK: test.format_custom_directive_successors ^bb1, ^bb2
  test.format_custom_directive_successors ^bb1, ^bb2

^bb1:
  // CHECK: test.format_custom_directive_successors ^bb2
  test.format_custom_directive_successors ^bb2

^bb2:
  return
}

//===----------------------------------------------------------------------===//
// Format trait type inference
//===----------------------------------------------------------------------===//

// CHECK: test.format_infer_variadic_type_from_non_variadic %[[I64]], %[[I64]] : i64
test.format_infer_variadic_type_from_non_variadic %i64, %i64 : i64

//===----------------------------------------------------------------------===//
// AllTypesMatch type inference
//===----------------------------------------------------------------------===//

// CHECK: test.format_all_types_match_var %[[I64]], %[[I64]] : i64
%ignored_res1 = test.format_all_types_match_var %i64, %i64 : i64

// CHECK: test.format_all_types_match_attr 1 : i64, %[[I64]]
%ignored_res2 = test.format_all_types_match_attr 1 : i64, %i64

//===----------------------------------------------------------------------===//
// TypesMatchWith type inference
//===----------------------------------------------------------------------===//

// CHECK: test.format_types_match_var %[[I64]] : i64
%ignored_res3 = test.format_types_match_var %i64 : i64

// CHECK: test.format_types_match_attr 1 : i64
%ignored_res4 = test.format_types_match_attr 1 : i64
