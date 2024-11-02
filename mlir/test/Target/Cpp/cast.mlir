// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s
// CHECK-LABEL: void cast
func.func @cast(%arg0 : i32) {
  // CHECK-NEXT: uint32_t [[V1:[^ ]*]] = (uint32_t) [[V0:[^ ]*]]
  %1 = emitc.cast %arg0: i32 to ui32

  // CHECK-NEXT: int64_t [[V4:[^ ]*]] = (int64_t) [[V0:[^ ]*]]
  %2 = emitc.cast %arg0: i32 to i64
  // CHECK-NEXT: int64_t [[V5:[^ ]*]] = (uint64_t) [[V0:[^ ]*]]
  %3 = emitc.cast %arg0: i32 to ui64

  // CHECK-NEXT: float [[V4:[^ ]*]] = (float) [[V0:[^ ]*]]
  %4 = emitc.cast %arg0: i32 to f32
  // CHECK-NEXT: double [[V5:[^ ]*]] = (double) [[V0:[^ ]*]]
  %5 = emitc.cast %arg0: i32 to f64

  // CHECK-NEXT: bool [[V6:[^ ]*]] = (bool) [[V0:[^ ]*]]
  %6 = emitc.cast %arg0: i32 to i1

  // CHECK-NEXT: mytype [[V7:[^ ]*]] = (mytype) [[V0:[^ ]*]]
  %7 = emitc.cast %arg0: i32 to !emitc.opaque<"mytype">
  return
}

// CHECK-LABEL: void cast_ptr
func.func @cast_ptr(%arg0 : !emitc.ptr<!emitc.opaque<"void">>) {
  // CHECK-NEXT: int32_t* [[V1:[^ ]*]] = (int32_t*) [[V0:[^ ]*]]
  %1 = emitc.cast %arg0 : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32>
  return
}
