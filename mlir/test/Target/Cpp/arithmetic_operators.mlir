// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s

func.func @add_int(%arg0: i32, %arg1: i32) {
  %1 = "emitc.add" (%arg0, %arg1) : (i32, i32) -> i32
  return
}
// CHECK-LABEL: void add_int
// CHECK-NEXT:  int32_t [[V2:[^ ]*]] = [[V0:[^ ]*]] + [[V1:[^ ]*]]

func.func @add_pointer(%arg0: !emitc.ptr<f32>, %arg1: i32) {
  %1 = "emitc.add" (%arg0, %arg1) : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
  return
}
// CHECK-LABEL: void add_pointer
// CHECK-NEXT:  float* [[V2:[^ ]*]] = [[V0:[^ ]*]] + [[V1:[^ ]*]]

func.func @sub_int(%arg0: i32, %arg1: i32) {
  %1 = "emitc.sub" (%arg0, %arg1) : (i32, i32) -> i32
  return
}
// CHECK-LABEL: void sub_int
// CHECK-NEXT:  int32_t [[V2:[^ ]*]] = [[V0:[^ ]*]] - [[V1:[^ ]*]]

func.func @sub_pointer(%arg0: !emitc.ptr<f32>, %arg1: !emitc.ptr<f32>) {
  %1 = "emitc.sub" (%arg0, %arg1) : (!emitc.ptr<f32>, !emitc.ptr<f32>) -> !emitc.opaque<"ptrdiff_t">
  return
}
// CHECK-LABEL: void sub_pointer
// CHECK-NEXT:  ptrdiff_t [[V2:[^ ]*]] = [[V0:[^ ]*]] - [[V1:[^ ]*]]
