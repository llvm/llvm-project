// RUN: mlir-opt %s -test-dialect-conversion-pdll | FileCheck %s

// CHECK-LABEL: @TestSingleConversion
func.func @TestSingleConversion() {
  // CHECK: %[[CAST:.*]] = "test.cast"() : () -> f64
  // CHECK-NEXT: "test.return"(%[[CAST]]) : (f64) -> ()
  %result = "test.cast"() : () -> (i64)
  "test.return"(%result) : (i64) -> ()
}

// CHECK-LABEL: @TestLingeringConversion
func.func @TestLingeringConversion() -> i64 {
  // CHECK: %[[ORIG_CAST:.*]] = "test.cast"() : () -> f64
  // CHECK: %[[MATERIALIZE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ORIG_CAST]] : f64 to i64
  // CHECK-NEXT: return %[[MATERIALIZE_CAST]] : i64
  %result = "test.cast"() : () -> (i64)
  return %result : i64
}
