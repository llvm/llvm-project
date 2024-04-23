// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s

func.func @logical(%arg0: i32, %arg1: i32) -> () {
  %0 = emitc.logical_and %arg0, %arg1 : i32, i32
  %1 = emitc.logical_not %arg0  : i32
  %2 = emitc.logical_or %arg0, %arg1 : i32, i32

  return
}

// CHECK-LABEL: void logical
// CHECK-NEXT:  bool [[V2:[^ ]*]] = [[V0:[^ ]*]] && [[V1:[^ ]*]];
// CHECK-NEXT:  bool [[V3:[^ ]*]] = ![[V0]];
// CHECK-NEXT:  bool [[V4:[^ ]*]] = [[V0]] || [[V1]];
