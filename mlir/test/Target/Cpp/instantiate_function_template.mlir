// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT
// RUN: mlir-translate -mlir-to-cpp -declare-variables-at-top %s | FileCheck %s -check-prefix=CPP-DECLTOP

func.func @emitc_instantiate_template() {
  %c1 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
  %0 = emitc.instantiate_function_template "func_template"<%c1> : (i32) -> !emitc.ptr<!emitc.opaque<"void">>
  return
}
// CPP-DEFAULT: void emitc_instantiate_template() {
// CPP-DEFAULT-NEXT: int32_t [[V0:[^ ]*]] = 7;
// CPP-DEFAULT-NEXT: void* [[V1:[^ ]*]] = &func_template<decltype([[V0]])>;

// CPP-DECLTOP: void emitc_instantiate_template() {
// CPP-DECLTOP-NEXT: int32_t [[V0:[^ ]*]];
// CPP-DECLTOP-NEXT: void* [[V1:[^ ]*]];
// CPP-DECLTOP-NEXT: [[V0]] = 7;
// CPP-DECLTOP-NEXT: [[V1]] = &func_template<decltype([[V0]])>;
