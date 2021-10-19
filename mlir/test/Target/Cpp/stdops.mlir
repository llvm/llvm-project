// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT
// RUN: mlir-translate -mlir-to-cpp -declare-variables-at-top %s | FileCheck %s -check-prefix=CPP-DECLTOP

func @std_constant() {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 2 : index
  %c2 = arith.constant 2.0 : f32
  %c3 = arith.constant dense<0> : tensor<i32>
  %c4 = arith.constant dense<[0, 1]> : tensor<2xindex>
  %c5 = arith.constant dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf32>
  return
}
// CPP-DEFAULT: void std_constant() {
// CPP-DEFAULT-NEXT: int32_t [[V0:[^ ]*]] = 0;
// CPP-DEFAULT-NEXT: size_t [[V1:[^ ]*]] = 2;
// CPP-DEFAULT-NEXT: float [[V2:[^ ]*]] = (float)2.000000000e+00;
// CPP-DEFAULT-NEXT: Tensor<int32_t> [[V3:[^ ]*]] = {0};
// CPP-DEFAULT-NEXT: Tensor<size_t, 2> [[V4:[^ ]*]] = {0, 1};
// CPP-DEFAULT-NEXT: Tensor<float, 2, 2> [[V5:[^ ]*]] = {(float)0.0e+00, (float)1.000000000e+00, (float)2.000000000e+00, (float)3.000000000e+00};

// CPP-DECLTOP: void std_constant() {
// CPP-DECLTOP-NEXT: int32_t [[V0:[^ ]*]];
// CPP-DECLTOP-NEXT: size_t [[V1:[^ ]*]];
// CPP-DECLTOP-NEXT: float [[V2:[^ ]*]];
// CPP-DECLTOP-NEXT: Tensor<int32_t> [[V3:[^ ]*]];
// CPP-DECLTOP-NEXT: Tensor<size_t, 2> [[V4:[^ ]*]];
// CPP-DECLTOP-NEXT: Tensor<float, 2, 2> [[V5:[^ ]*]];
// CPP-DECLTOP-NEXT: [[V0]] = 0;
// CPP-DECLTOP-NEXT: [[V1]] = 2;
// CPP-DECLTOP-NEXT: [[V2]] = (float)2.000000000e+00;
// CPP-DECLTOP-NEXT: [[V3]] = {0};
// CPP-DECLTOP-NEXT: [[V4]] = {0, 1};
// CPP-DECLTOP-NEXT: [[V5]] = {(float)0.0e+00, (float)1.000000000e+00, (float)2.000000000e+00, (float)3.000000000e+00};

func @std_call() {
  %0 = call @one_result () : () -> i32
  %1 = call @one_result () : () -> i32
  return
}
// CPP-DEFAULT: void std_call() {
// CPP-DEFAULT-NEXT: int32_t [[V0:[^ ]*]] = one_result();
// CPP-DEFAULT-NEXT: int32_t [[V1:[^ ]*]] = one_result();

// CPP-DECLTOP: void std_call() {
// CPP-DECLTOP-NEXT: int32_t [[V0:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[V1:[^ ]*]];
// CPP-DECLTOP-NEXT: [[V0]] = one_result();
// CPP-DECLTOP-NEXT: [[V1]] = one_result();


func @std_call_two_results() {
  %c = arith.constant 0 : i8
  %0:2 = call @two_results () : () -> (i32, f32)
  %1:2 = call @two_results () : () -> (i32, f32)
  return
}
// CPP-DEFAULT: void std_call_two_results() {
// CPP-DEFAULT-NEXT: int8_t  [[V0:[^ ]*]] = 0;
// CPP-DEFAULT-NEXT: int32_t [[V1:[^ ]*]];
// CPP-DEFAULT-NEXT: float [[V2:[^ ]*]];
// CPP-DEFAULT-NEXT: std::tie([[V1]], [[V2]]) = two_results();
// CPP-DEFAULT-NEXT: int32_t [[V3:[^ ]*]];
// CPP-DEFAULT-NEXT: float [[V4:[^ ]*]];
// CPP-DEFAULT-NEXT: std::tie([[V3]], [[V4]]) = two_results();

// CPP-DECLTOP: void std_call_two_results() {
// CPP-DECLTOP-NEXT: int8_t [[V0:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[V1:[^ ]*]];
// CPP-DECLTOP-NEXT: float [[V2:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[V3:[^ ]*]];
// CPP-DECLTOP-NEXT: float [[V4:[^ ]*]];
// CPP-DECLTOP-NEXT: [[V0]] = 0;
// CPP-DECLTOP-NEXT: std::tie([[V1]], [[V2]]) = two_results();
// CPP-DECLTOP-NEXT: std::tie([[V3]], [[V4]]) = two_results();


func @one_result() -> i32 {
  %0 = arith.constant 0 : i32
  return %0 : i32
}
// CPP-DEFAULT: int32_t one_result() {
// CPP-DEFAULT-NEXT: int32_t [[V0:[^ ]*]] = 0;
// CPP-DEFAULT-NEXT: return [[V0]];

// CPP-DECLTOP: int32_t one_result() {
// CPP-DECLTOP-NEXT: int32_t [[V0:[^ ]*]];
// CPP-DECLTOP-NEXT: [[V0]] = 0;
// CPP-DECLTOP-NEXT: return [[V0]];


func @two_results() -> (i32, f32) {
  %0 = arith.constant 0 : i32
  %1 = arith.constant 1.0 : f32
  return %0, %1 : i32, f32
}
// CPP-DEFAULT: std::tuple<int32_t, float> two_results() {
// CPP-DEFAULT: int32_t [[V0:[^ ]*]] = 0;
// CPP-DEFAULT: float [[V1:[^ ]*]] = (float)1.000000000e+00;
// CPP-DEFAULT: return std::make_tuple([[V0]], [[V1]]);

// CPP-DECLTOP: std::tuple<int32_t, float> two_results() {
// CPP-DECLTOP: int32_t [[V0:[^ ]*]];
// CPP-DECLTOP: float [[V1:[^ ]*]];
// CPP-DECLTOP: [[V0]] = 0;
// CPP-DECLTOP: [[V1]] = (float)1.000000000e+00;
// CPP-DECLTOP: return std::make_tuple([[V0]], [[V1]]);


func @single_return_statement(%arg0 : i32) -> i32 {
  return %arg0 : i32
}
// CPP-DEFAULT: int32_t single_return_statement(int32_t [[V0:[^ ]*]]) {
// CPP-DEFAULT-NEXT: return [[V0]];

// CPP-DECLTOP: int32_t single_return_statement(int32_t [[V0:[^ ]*]]) {
// CPP-DECLTOP-NEXT: return [[V0]];
