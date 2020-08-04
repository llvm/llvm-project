// RUN: mlir-opt -allow-unregistered-dialect %s | mlir-opt -allow-unregistered-dialect -verify-diagnostics | FileCheck %s

// CHECK: %[[I64:.*]] =
%i64 = "foo.op"() : () -> (i64)
// CHECK: %[[MEMREF:.*]] =
%memref = "foo.op"() : () -> (memref<1xf64>)

// CHECK: test.format_literal_op keyword_$. -> :, = <> () [] {foo.some_attr}
test.format_literal_op keyword_$. -> :, = <> () [] {foo.some_attr}

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

// CHECK: test.format_attr_dict_w_keyword attributes {attr = 10 : i64}
test.format_attr_dict_w_keyword attributes {attr = 10 : i64}

// CHECK: test.format_attr_dict_w_keyword attributes {attr = 10 : i64, opt_attr = 10 : i64}
test.format_attr_dict_w_keyword attributes {attr = 10 : i64, opt_attr = 10 : i64}

// CHECK: test.format_buildable_type_op %[[I64]]
%ignored = test.format_buildable_type_op %i64

//===----------------------------------------------------------------------===//
// Format results
//===----------------------------------------------------------------------===//

// CHECK: test.format_result_a_op memref<1xf64>
%ignored_a:2 = test.format_result_a_op memref<1xf64>

// CHECK: test.format_result_b_op i64, memref<1xf64>
%ignored_b:2 = test.format_result_b_op i64, memref<1xf64>

// CHECK: test.format_result_c_op (i64) -> memref<1xf64>
%ignored_c:2 = test.format_result_c_op (i64) -> memref<1xf64>

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
