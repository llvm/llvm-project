// RUN: mlir-opt %s  --verify-roundtrip

func.func @deeply_nested_attribute() -> i32 {
  %0 = "test.constant"() {value = [[[[[[[[[[[[1]]]]]]]]]]]]} : () -> i32
  return %0 : i32
}