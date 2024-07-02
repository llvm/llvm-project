// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s
// RUN: mlir-translate -mlir-to-cpp -declare-variables-at-top %s | FileCheck %s

func.func @load_store_array(%arg0: !emitc.array<4x8xf32>, %arg1: !emitc.array<3x5xf32>, %arg2: index, %arg3: index) {
  %0 = emitc.subscript %arg0[%arg2, %arg3] : (!emitc.array<4x8xf32>, index, index) -> f32
  %1 = emitc.subscript %arg1[%arg2, %arg3] : (!emitc.array<3x5xf32>, index, index) -> f32
  emitc.assign %0 : f32 to %1 : f32
  return
}
// CHECK: void load_store_array(float [[ARR1:[^ ]*]][4][8], float [[ARR2:[^ ]*]][3][5],
// CHECK-SAME:            size_t [[I:[^ ]*]], size_t [[J:[^ ]*]])
// CHECK-NEXT: [[ARR2]][[[I]]][[[J]]] = [[ARR1]][[[I]]][[[J]]];

func.func @load_store_pointer(%arg0: !emitc.ptr<f32>, %arg1: !emitc.ptr<f32>, %arg2: index, %arg3: index) {
  %0 = emitc.subscript %arg0[%arg2] : (!emitc.ptr<f32>, index) -> f32
  %1 = emitc.subscript %arg1[%arg3] : (!emitc.ptr<f32>, index) -> f32
  emitc.assign %0 : f32 to %1 : f32
  return
}
// CHECK: void load_store_pointer(float* [[PTR1:[^ ]*]], float* [[PTR2:[^ ]*]],
// CHECK-SAME:            size_t [[I:[^ ]*]], size_t [[J:[^ ]*]])
// CHECK-NEXT: [[PTR2]][[[J]]] = [[PTR1]][[[I]]];

func.func @load_store_opaque(%arg0: !emitc.opaque<"std::map<char, int>">, %arg1: !emitc.opaque<"std::map<char, int>">, %arg2: !emitc.opaque<"char">, %arg3: !emitc.opaque<"char">) {
  %0 = emitc.subscript %arg0[%arg2] : (!emitc.opaque<"std::map<char, int>">, !emitc.opaque<"char">) -> !emitc.opaque<"int">
  %1 = emitc.subscript %arg1[%arg3] : (!emitc.opaque<"std::map<char, int>">, !emitc.opaque<"char">) -> !emitc.opaque<"int">
  emitc.assign %0 : !emitc.opaque<"int"> to %1 : !emitc.opaque<"int">
  return
}
// CHECK: void load_store_opaque(std::map<char, int> [[MAP1:[^ ]*]], std::map<char, int> [[MAP2:[^ ]*]],
// CHECK-SAME:            char [[I:[^ ]*]], char [[J:[^ ]*]])
// CHECK-NEXT: [[MAP2]][[[J]]] = [[MAP1]][[[I]]];

emitc.func @func1(%arg0 : f32) {
  emitc.return
}

emitc.func @call_arg(%arg0: !emitc.array<4x8xf32>, %i: i32, %j: i16,
                     %k: i8) {
  %0 = emitc.subscript %arg0[%i, %j] : (!emitc.array<4x8xf32>, i32, i16) -> f32
  %1 = emitc.subscript %arg0[%j, %k] : (!emitc.array<4x8xf32>, i16, i8) -> f32

  emitc.call @func1 (%0) : (f32) -> ()
  emitc.call_opaque "func2" (%1) : (f32) -> ()
  emitc.call_opaque "func3" (%0, %1) { args = [1 : index, 0 : index] } : (f32, f32) -> ()
  emitc.return
}
// CHECK: void call_arg(float [[ARR1:[^ ]*]][4][8], int32_t [[I:[^ ]*]],
// CHECK-SAME:          int16_t [[J:[^ ]*]], int8_t [[K:[^ ]*]])
// CHECK-NEXT: func1([[ARR1]][[[I]]][[[J]]]);
// CHECK-NEXT: func2([[ARR1]][[[J]]][[[K]]]);
// CHECK-NEXT: func3([[ARR1]][[[J]]][[[K]]], [[ARR1]][[[I]]][[[J]]]);
