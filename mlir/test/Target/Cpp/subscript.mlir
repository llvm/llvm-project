// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s
// RUN: mlir-translate -mlir-to-cpp -declare-variables-at-top %s | FileCheck %s

func.func @load_store(%arg0: !emitc.array<4x8xf32>, %arg1: !emitc.array<3x5xf32>, %arg2: index, %arg3: index) {
  %0 = emitc.subscript %arg0[%arg2, %arg3] : <4x8xf32>
  %1 = emitc.subscript %arg1[%arg2, %arg3] : <3x5xf32>
  emitc.assign %0 : f32 to %1 : f32
  return
}
// CHECK: void load_store(float [[ARR1:[^ ]*]][4][8], float [[ARR2:[^ ]*]][3][5],
// CHECK-SAME:            size_t [[I:[^ ]*]], size_t [[J:[^ ]*]])
// CHECK-NEXT: [[ARR2]][[[I]]][[[J]]] = [[ARR1]][[[I]]][[[J]]];
