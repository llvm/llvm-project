// RUN: mlir-opt %s -test-remapped-value | FileCheck %s

// Simple test that exercises ConvertPatternRewriter::getRemappedValue.

// CHECK-LABEL: func @remap_input_1_to_1
// CHECK-SAME: (%[[ARG:.*]]: i32)
// CHECK-NEXT: %[[VAL:.*]] = "test.one_variadic_out_one_variadic_in1"(%[[ARG]], %[[ARG]])
// CHECK-NEXT: "test.one_variadic_out_one_variadic_in1"(%[[VAL]], %[[VAL]])

func @remap_input_1_to_1(%arg0: i32) {
  %0 = "test.one_variadic_out_one_variadic_in1"(%arg0) : (i32) -> i32
  %1 = "test.one_variadic_out_one_variadic_in1"(%0) : (i32) -> i32
  "test.return"() : () -> ()
}

// Test the case where an operation is converted before its operands are.

// CHECK-LABEL: func @remap_unconverted
// CHECK-NEXT: %[[VAL:.*]] = "test.type_producer"() : () -> f64
// CHECK-NEXT: "test.type_consumer"(%[[VAL]]) : (f64)
func @remap_unconverted() {
  %region_result = "test.remapped_value_region"() ({
    %result = "test.type_producer"() : () -> f32
    "test.return"(%result) : (f32) -> ()
  }) : () -> (f32)
  "test.type_consumer"(%region_result) : (f32) -> ()
  "test.return"() : () -> ()
}
