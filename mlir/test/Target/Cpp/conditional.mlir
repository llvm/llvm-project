// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s

func.func @cond(%cond: i1, %arg0: i32, %arg1: i32) -> () {
  %0 = emitc.conditional %cond, %arg0, %arg1 : i32
  return
}

// CHECK-LABEL: void cond
// CHECK-NEXT:  int32_t [[V3:[^ ]*]] = [[V0:[^ ]*]] ? [[V1:[^ ]*]] : [[V2:[^ ]*]];
