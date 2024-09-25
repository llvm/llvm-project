// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s

func.func @unary(%arg0: i32) -> () {
  %0 = emitc.unary_minus %arg0 : (i32) -> i32
  %1 = emitc.unary_plus %arg0 : (i32) -> i32

  return
}

// CHECK-LABEL: void unary
// CHECK-NEXT:  int32_t [[V1:[^ ]*]] = -[[V0:[^ ]*]];
// CHECK-NEXT:  int32_t [[V2:[^ ]*]] = +[[V0]];
