// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s

func.func @bitwise(%arg0: i32, %arg1: i32) -> () {
  %0 = emitc.bitwise_and %arg0, %arg1 : (i32, i32) -> i32
  %1 = emitc.bitwise_left_shift %arg0, %arg1 : (i32, i32) -> i32
  %2 = emitc.bitwise_not %arg0 : (i32) -> i32
  %3 = emitc.bitwise_or %arg0, %arg1 : (i32, i32) -> i32
  %4 = emitc.bitwise_right_shift %arg0, %arg1 : (i32, i32) -> i32
  %5 = emitc.bitwise_xor %arg0, %arg1 : (i32, i32) -> i32

  return
}

// CHECK-LABEL: void bitwise
// CHECK-NEXT:  int32_t [[V2:[^ ]*]] = [[V0:[^ ]*]] & [[V1:[^ ]*]];
// CHECK-NEXT:  int32_t [[V3:[^ ]*]] = [[V0]] << [[V1]];
// CHECK-NEXT:  int32_t [[V4:[^ ]*]] = ~[[V0]];
// CHECK-NEXT:  int32_t [[V5:[^ ]*]] = [[V0]] | [[V1]];
// CHECK-NEXT:  int32_t [[V6:[^ ]*]] = [[V0]] >> [[V1]];
// CHECK-NEXT:  int32_t [[V7:[^ ]*]] = [[V0]] ^ [[V1]];
