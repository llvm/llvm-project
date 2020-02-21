// RUN: mlir-opt %s | mlir-opt -verify-diagnostics | FileCheck %s

// CHECK: %[[I64:.*]] =
%i64 = "foo.op"() : () -> (i64)
// CHECK: %[[MEMREF:.*]] =
%memref = "foo.op"() : () -> (memref<1xf64>)

// CHECK: test.format_literal_op keyword_$. -> :, = <> () [] {foo.some_attr}
test.format_literal_op keyword_$. -> :, = <> () [] {foo.some_attr}

// CHECK: test.format_attr_op 10
// CHECK-NOT: {attr
test.format_attr_op 10

// CHECK: test.format_attr_dict_w_keyword attributes {attr = 10 : i64}
test.format_attr_dict_w_keyword attributes {attr = 10 : i64}

// CHECK: test.format_buildable_type_op %[[I64]]
%ignored = test.format_buildable_type_op %i64

// CHECK: test.format_result_a_op memref<1xf64>
%ignored_a:2 = test.format_result_a_op memref<1xf64>

// CHECK: test.format_result_b_op i64, memref<1xf64>
%ignored_b:2 = test.format_result_b_op i64, memref<1xf64>

// CHECK: test.format_result_c_op (i64) -> memref<1xf64>
%ignored_c:2 = test.format_result_c_op (i64) -> memref<1xf64>

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
