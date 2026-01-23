// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s --match-full-lines
// RUN: mlir-translate -mlir-to-cpp -declare-variables-at-top %s | FileCheck %s --match-full-lines


emitc.verbatim "#ifdef __cplusplus"
// CHECK: #ifdef __cplusplus
emitc.verbatim "extern \"C\" {"
// CHECK-NEXT: extern "C" {
emitc.verbatim "#endif  // __cplusplus"
// CHECK-NEXT: #endif  // __cplusplus
emitc.verbatim "#ifdef __cplusplus"
// CHECK-NEXT: #ifdef __cplusplus
emitc.verbatim "}  // extern \"C\""
// CHECK-NEXT: }  // extern "C"
emitc.verbatim "#endif  // __cplusplus"
// CHECK-NEXT: #endif  // __cplusplus

emitc.verbatim "typedef int32_t i32;"
// CHECK-NEXT: typedef int32_t i32;
emitc.verbatim "typedef float f32;"
// CHECK-NEXT: typedef float f32;

emitc.func @func(%arg: f32) {
  // CHECK: void func(float [[V0:[^ ]*]]) {
  %a = "emitc.variable"(){value = #emitc.opaque<"">} : () -> !emitc.array<3x7xi32>
  // CHECK: int32_t [[A:[^ ]*]][3][7];

  emitc.verbatim "{}" args %arg : f32
  // CHECK: [[V0]]

  emitc.verbatim "{} {{a" args %arg : f32
  // CHECK-NEXT: [[V0]] {a

  emitc.verbatim "#pragma my var={} property" args %arg : f32
  // CHECK-NEXT: #pragma my var=[[V0]] property

  emitc.verbatim "#pragma my2 var={} property" args %a : !emitc.array<3x7xi32>
  // CHECK-NEXT: #pragma my2 var=[[A]] property

  emitc.return
}
