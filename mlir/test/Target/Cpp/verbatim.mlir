// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s
// RUN: mlir-translate -mlir-to-cpp -declare-variables-at-top %s | FileCheck %s


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

emitc.verbatim "typedef int32_t i32" {trailing_semicolon = unit}
// CHECK-NEXT: typedef int32_t i32;
emitc.verbatim "typedef float f32" trailing_semicolon
// CHECK-NEXT: typedef float f32;
