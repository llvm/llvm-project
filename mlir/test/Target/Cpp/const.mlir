// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT
// RUN: mlir-translate -mlir-to-cpp -declare-variables-at-top %s | FileCheck %s -check-prefix=CPP-DECLTOP

func.func @emitc_constant() {
  %c0 = "emitc.constant"(){value = #emitc.opaque<"INT_MAX">} : () -> i32
  %c1 = "emitc.constant"(){value = 42 : i32} : () -> i32
  %c2 = "emitc.constant"(){value = -1 : i32} : () -> i32
  %c3 = "emitc.constant"(){value = -1 : si8} : () -> si8
  %c4 = "emitc.constant"(){value = 255 : ui8} : () -> ui8
  %c5 = "emitc.constant"(){value = #emitc.opaque<"CHAR_MIN">} : () -> !emitc.opaque<"char">
  %c6 = "emitc.constant"(){value = 2 : index} : () -> index
  %c7 = "emitc.constant"(){value = 2.0 : f32} : () -> f32
  %f64 = "emitc.constant"(){value = 4.0 : f64} : () -> f64
  %f16 = "emitc.constant"(){value = 2.0 : f16} : () -> f16
  %bf16 = "emitc.constant"(){value = 4.0 : bf16} : () -> bf16
  %c8 = "emitc.constant"(){value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %c9 = "emitc.constant"(){value = dense<[0, 1]> : tensor<2xindex>} : () -> tensor<2xindex>
  %c10 = "emitc.constant"(){value = dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
  %c11 = "emitc.constant"(){value = dense<[0, 1]> : !emitc.array<2xindex>} : () -> !emitc.array<2xindex>
  %c12 = "emitc.constant"(){value = dense<[0.0, 1.0]> : !emitc.array<2xf32>} : () -> !emitc.array<2xf32>
  return
}
// CPP-DEFAULT: void emitc_constant() {
// CPP-DEFAULT-NEXT: int32_t [[V0:[^ ]*]] = INT_MAX;
// CPP-DEFAULT-NEXT: int32_t [[V1:[^ ]*]] = 42;
// CPP-DEFAULT-NEXT: int32_t [[V2:[^ ]*]] = -1;
// CPP-DEFAULT-NEXT: int8_t [[V3:[^ ]*]] = -1;
// CPP-DEFAULT-NEXT: uint8_t [[V4:[^ ]*]] = 255;
// CPP-DEFAULT-NEXT: char [[V5:[^ ]*]] = CHAR_MIN;
// CPP-DEFAULT-NEXT: size_t [[V6:[^ ]*]] = 2;
// CPP-DEFAULT-NEXT: float [[V7:[^ ]*]] = 2.000000000e+00f;
// CPP-DEFAULT-NEXT: double [[F64:[^ ]*]] = 4.00000000000000000e+00;
// CPP-DEFAULT-NEXT: _Float16 [[F16:[^ ]*]] = 2.00000e+00f16;
// CPP-DEFAULT-NEXT: __bf16 [[BF16:[^ ]*]] = 4.0000e+00bf16;
// CPP-DEFAULT-NEXT: Tensor<int32_t> [[V8:[^ ]*]] = {0};
// CPP-DEFAULT-NEXT: Tensor<size_t, 2> [[V9:[^ ]*]] = {0, 1};
// CPP-DEFAULT-NEXT: Tensor<float, 2, 2> [[V10:[^ ]*]] = {0.0e+00f, 1.000000000e+00f, 2.000000000e+00f, 3.000000000e+00f};
// CPP-DEFAULT-NEXT: size_t [[V11:[^ ]*]][2] = {0, 1};
// CPP-DEFAULT-NEXT: float [[V12:[^ ]*]][2] = {0.0e+00f, 1.000000000e+00f};

// CPP-DECLTOP: void emitc_constant() {
// CPP-DECLTOP-NEXT: int32_t [[V0:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[V1:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[V2:[^ ]*]];
// CPP-DECLTOP-NEXT: int8_t [[V3:[^ ]*]];
// CPP-DECLTOP-NEXT: uint8_t [[V4:[^ ]*]];
// CPP-DECLTOP-NEXT: char [[V5:[^ ]*]];
// CPP-DECLTOP-NEXT: size_t [[V6:[^ ]*]];
// CPP-DECLTOP-NEXT: float [[V7:[^ ]*]];
// CPP-DECLTOP-NEXT: double [[F64:[^ ]*]];
// CPP-DECLTOP-NEXT: _Float16 [[F16:[^ ]*]];
// CPP-DECLTOP-NEXT: __bf16 [[BF16:[^ ]*]];
// CPP-DECLTOP-NEXT: Tensor<int32_t> [[V8:[^ ]*]];
// CPP-DECLTOP-NEXT: Tensor<size_t, 2> [[V9:[^ ]*]];
// CPP-DECLTOP-NEXT: Tensor<float, 2, 2> [[V10:[^ ]*]];
// CPP-DECLTOP-NEXT: size_t [[V11:[^ ]*]][2];
// CPP-DECLTOP-NEXT: float [[V12:[^ ]*]][2];
// CPP-DECLTOP-NEXT: [[V0]] = INT_MAX;
// CPP-DECLTOP-NEXT: [[V1]] = 42;
// CPP-DECLTOP-NEXT: [[V2]] = -1;
// CPP-DECLTOP-NEXT: [[V3]] = -1;
// CPP-DECLTOP-NEXT: [[V4]] = 255;
// CPP-DECLTOP-NEXT: [[V5]] = CHAR_MIN;
// CPP-DECLTOP-NEXT: [[V6]] = 2;
// CPP-DECLTOP-NEXT: [[V7]] = 2.000000000e+00f;
// CPP-DECLTOP-NEXT: [[F64]] = 4.00000000000000000e+00;
// CPP-DECLTOP-NEXT: [[F16]] = 2.00000e+00f16;
// CPP-DECLTOP-NEXT: [[BF16]] = 4.0000e+00bf16;
// CPP-DECLTOP-NEXT: [[V8]] = {0};
// CPP-DECLTOP-NEXT: [[V9]] = {0, 1};
// CPP-DECLTOP-NEXT: [[V10]] = {0.0e+00f, 1.000000000e+00f, 2.000000000e+00f, 3.000000000e+00f};
// CPP-DECLTOP-NEXT: [[V11]] = {0, 1};
// CPP-DECLTOP-NEXT: [[V12]] = {0.0e+00f, 1.000000000e+00f};
