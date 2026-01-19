// RUN: mlir-opt %s --verify-roundtrip  --split-input-file

func.func @deeply_nested_attribute() -> i32 {
  %0 = "test.constant"() {value = [[[[[[[[[[[[1]]]]]]]]]]]]} : () -> i32
  return %0 : i32
}

// -----
// Make sure bytecode reader doesn't get stuck in an infinite loop when
// parsing a specific deeply nested attribute structure.
func.func @deeply_nested_infinite_loop() -> i32 {
  %0 = "test.constant"() {value = [[[[[[1 : i64]]]]]]} : () -> i32
  return %0 : i32
}
