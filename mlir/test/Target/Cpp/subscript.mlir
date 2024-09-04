// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT
// RUN: mlir-translate -mlir-to-cpp -declare-variables-at-top %s | FileCheck %s -check-prefix=CPP-DECLTOP

func.func @load_store_array(%arg0: !emitc.array<4x8xf32>, %arg1: !emitc.array<3x5xf32>, %arg2: index, %arg3: index) {
  %0 = emitc.subscript %arg0[%arg2, %arg3] : (!emitc.array<4x8xf32>, index, index) -> !emitc.lvalue<f32>
  %1 = emitc.subscript %arg1[%arg2, %arg3] : (!emitc.array<3x5xf32>, index, index) -> !emitc.lvalue<f32>
  %2 = emitc.load %0 : <f32>
  emitc.assign %2 : f32 to %1 : !emitc.lvalue<f32>
  return
}
// CPP-DEFAULT: void load_store_array(float [[ARR1:[^ ]*]][4][8], float [[ARR2:[^ ]*]][3][5],
// CPP-DEFAULT-SAME:            size_t [[I:[^ ]*]], size_t [[J:[^ ]*]])
// CPP-DEFAULT-NEXT: float [[VAL:[^ ]*]] = [[ARR1]][[[I]]][[[J]]];
// CPP-DEFAULT-NEXT: [[ARR2]][[[I]]][[[J]]] = [[VAL]];

// CPP-DECLTOP: void load_store_array(float [[ARR1:[^ ]*]][4][8], float [[ARR2:[^ ]*]][3][5],
// CPP-DECLTOP-SAME:            size_t [[I:[^ ]*]], size_t [[J:[^ ]*]])
// CPP-DECLTOP-NEXT: float [[VAL:[^ ]*]];
// CPP-DECLTOP-NEXT: [[VAL]] = [[ARR1]][[[I]]][[[J]]];
// CPP-DECLTOP-NEXT: [[ARR2]][[[I]]][[[J]]] = [[VAL]];

func.func @load_store_pointer(%arg0: !emitc.ptr<f32>, %arg1: !emitc.ptr<f32>, %arg2: index, %arg3: index) {
  %0 = emitc.subscript %arg0[%arg2] : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
  %1 = emitc.subscript %arg1[%arg3] : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
  %2 = emitc.load %0 : <f32>
  emitc.assign %2 : f32 to %1 : <f32>
  return
}
// CPP-DEFAULT: void load_store_pointer(float* [[PTR1:[^ ]*]], float* [[PTR2:[^ ]*]],
// CPP-DEFAULT-SAME:            size_t [[I:[^ ]*]], size_t [[J:[^ ]*]])
// CPP-DEFAULT-NEXT: float [[VAL:[^ ]*]] = [[PTR1]][[[I]]];
// CPP-DEFAULT-NEXT: [[PTR2]][[[J]]] = [[VAL]];

// CPP-DECLTOP: void load_store_pointer(float* [[PTR1:[^ ]*]], float* [[PTR2:[^ ]*]],
// CPP-DECLTOP-SAME:            size_t [[I:[^ ]*]], size_t [[J:[^ ]*]])
// CPP-DECLTOP-NEXT: float [[VAL:[^ ]*]];
// CPP-DECLTOP-NEXT: [[VAL]] = [[PTR1]][[[I]]];
// CPP-DECLTOP-NEXT: [[PTR2]][[[J]]] = [[VAL]];

func.func @load_store_opaque(%arg0: !emitc.opaque<"std::map<char, int>">, %arg1: !emitc.opaque<"std::map<char, int>">, %arg2: !emitc.opaque<"char">, %arg3: !emitc.opaque<"char">) {
  %0 = emitc.subscript %arg0[%arg2] : (!emitc.opaque<"std::map<char, int>">, !emitc.opaque<"char">) -> !emitc.lvalue<!emitc.opaque<"int">>
  %1 = emitc.subscript %arg1[%arg3] : (!emitc.opaque<"std::map<char, int>">, !emitc.opaque<"char">) -> !emitc.lvalue<!emitc.opaque<"int">>
  %2 = emitc.load %0 : <!emitc.opaque<"int">>
  emitc.assign %2 : !emitc.opaque<"int"> to %1 : <!emitc.opaque<"int">>
  return
}
// CPP-DEFAULT: void load_store_opaque(std::map<char, int> [[MAP1:[^ ]*]], std::map<char, int> [[MAP2:[^ ]*]],
// CPP-DEFAULT-SAME:            char [[I:[^ ]*]], char [[J:[^ ]*]])
// CPP-DEFAULT-NEXT: int [[VAL:[^ ]*]] = [[MAP1]][[[I]]];
// CPP-DEFAULT-NEXT: [[MAP2]][[[J]]] = [[VAL]];

// CPP-DECLTOP: void load_store_opaque(std::map<char, int> [[MAP1:[^ ]*]], std::map<char, int> [[MAP2:[^ ]*]],
// CPP-DECLTOP-SAME:            char [[I:[^ ]*]], char [[J:[^ ]*]])
// CPP-DECLTOP-NEXT: int [[VAL:[^ ]*]];
// CPP-DECLTOP-NEXT: [[VAL]] = [[MAP1]][[[I]]];
// CPP-DECLTOP-NEXT: [[MAP2]][[[J]]] = [[VAL]];

emitc.func @func1(%arg0 : f32) {
  emitc.return
}

emitc.func @call_arg(%arg0: !emitc.array<4x8xf32>, %i: i32, %j: i16,
                     %k: i8) {
  %0 = emitc.subscript %arg0[%i, %j] : (!emitc.array<4x8xf32>, i32, i16) -> !emitc.lvalue<f32>
  %1 = emitc.subscript %arg0[%j, %k] : (!emitc.array<4x8xf32>, i16, i8) -> !emitc.lvalue<f32>

  %2 = emitc.load %0 : <f32>
  emitc.call @func1 (%2) : (f32) -> ()
  %3 = emitc.load %1 : <f32>
  emitc.call_opaque "func2" (%3) : (f32) -> ()
  %4 = emitc.load %0 : <f32>
  %5 = emitc.load %1 : <f32>
  emitc.call_opaque "func3" (%4, %5) { args = [1 : index, 0 : index] } : (f32, f32) -> ()
  emitc.return
}
// CPP-DEFAULT: void call_arg(float [[ARR1:[^ ]*]][4][8], int32_t [[I:[^ ]*]],
// CPP-DEFAULT-SAME:          int16_t [[J:[^ ]*]], int8_t [[K:[^ ]*]])
// CPP-DEFAULT-NEXT: float [[VAL0:[^ ]*]] = [[ARR1]][[[I]]][[[J]]];
// CPP-DEFAULT-NEXT: func1([[VAL0]]);
// CPP-DEFAULT-NEXT: float [[VAL1:[^ ]*]] = [[ARR1]][[[J]]][[[K]]];
// CPP-DEFAULT-NEXT: func2([[VAL1]]);
// CPP-DEFAULT-NEXT: float [[VAL2:[^ ]*]] = [[ARR1]][[[I]]][[[J]]];
// CPP-DEFAULT-NEXT: float [[VAL3:[^ ]*]] = [[ARR1]][[[J]]][[[K]]];
// CPP-DEFAULT-NEXT: func3([[VAL3]], [[VAL2]]);

// CPP-DECLTOP: void call_arg(float [[ARR1:[^ ]*]][4][8], int32_t [[I:[^ ]*]],
// CPP-DECLTOP-SAME:          int16_t [[J:[^ ]*]], int8_t [[K:[^ ]*]])
// CPP-DECLTOP-NEXT: float [[VAL0:[^ ]*]];
// CPP-DECLTOP-NEXT: float [[VAL1:[^ ]*]];
// CPP-DECLTOP-NEXT: float [[VAL2:[^ ]*]];
// CPP-DECLTOP-NEXT: float [[VAL3:[^ ]*]];
// CPP-DECLTOP-NEXT: [[VAL0]] = [[ARR1]][[[I]]][[[J]]];
// CPP-DECLTOP-NEXT: func1([[VAL0]]);
// CPP-DECLTOP-NEXT: [[VAL1]] = [[ARR1]][[[J]]][[[K]]];
// CPP-DECLTOP-NEXT: func2([[VAL1]]);
// CPP-DECLTOP-NEXT: [[VAL2]] = [[ARR1]][[[I]]][[[J]]];
// CPP-DECLTOP-NEXT: [[VAL3]] = [[ARR1]][[[J]]][[[K]]];
// CPP-DECLTOP-NEXT: func3([[VAL3]], [[VAL2]]);
