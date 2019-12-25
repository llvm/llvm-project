// RUN: mlir-opt %s -test-remapped-value | FileCheck %s

// Simple test that exercises ConvertPatternRewriter::getRemappedValue.
func @remap_input_1_to_1(%arg0: i32) {
  %0 = "test.one_variadic_out_one_variadic_in1"(%arg0) : (i32) -> i32
  %1 = "test.one_variadic_out_one_variadic_in1"(%0) : (i32) -> i32
  "test.return"() : () -> ()
}
// CHECK-LABEL: func @remap_input_1_to_1
// CHECK-SAME: (%[[ARG:.*]]: i32)
// CHECK-NEXT: %[[VAL:.*]] = "test.one_variadic_out_one_variadic_in1"(%[[ARG]], %[[ARG]])
// CHECK-NEXT: "test.one_variadic_out_one_variadic_in1"(%[[VAL]], %[[VAL]])

