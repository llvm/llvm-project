// RUN: mlir-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s

#sg_map_fp16_a = #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 2]>
#sg_map_fp16_b = #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
#sg_map_fp16_c = #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
#sg_map_fp16_d = #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 2]>
// CHECK-LABEL: func @test_load_nd_fp16({{.*}}) {
func.func @test_load_nd_fp16(%A: memref<24x32xf16>, %B : memref<24x32xf16>, %C : memref<24x32xf16>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  // CHECK: xegpu.create_nd_tdesc %{{arg[0-9]}}[%{{c[0-9]+}}, %{{c[0-9]+}}] : memref<24x32xf16>
  // CHECK-SAME: -> !xegpu.tensor_desc<8x16xf16, #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 2]>>
  %1 = xegpu.create_nd_tdesc %A[%c0, %c1]
      : memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16, #sg_map_fp16_a>

  // CHECK: xegpu.load_nd %{{[0-9]}} {vnni_axis = 1 : i64}
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf16, #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 2]>> -> vector<4x1x2xf16>
  %2 = xegpu.load_nd %1 {vnni_axis = 1} : !xegpu.tensor_desc<8x16xf16, #sg_map_fp16_a> -> vector<4x1x2xf16>

  // CHECK: xegpu.create_nd_tdesc %{{arg[0-9]}}[%{{c[0-9]+}}, %{{c[0-9]+}}] : memref<24x32xf16>
  // CHECK-SAME: -> !xegpu.tensor_desc<16x16xf16, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>
  %3 = xegpu.create_nd_tdesc %B[%c0, %c1]
    : memref<24x32xf16> -> !xegpu.tensor_desc<16x16xf16, #sg_map_fp16_b>

  // CHECK: xegpu.load_nd %{{[0-9]}} {vnni_axis = 0 : i64}
  // CHECK-SAME: !xegpu.tensor_desc<16x16xf16, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>> -> vector<8x1x2xf16>
  %4 = xegpu.load_nd %3 {vnni_axis = 0} : !xegpu.tensor_desc<16x16xf16, #sg_map_fp16_b> -> vector<8x1x2xf16>

  // CHECK: xegpu.create_nd_tdesc %{{arg[0-9]}}[%{{c[0-9]+}}, %{{c[0-9]+}}] : memref<24x32xf16>
  // CHECK-SAME: -> !xegpu.tensor_desc<8x16xf32, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>
  %5 = xegpu.create_nd_tdesc %C[%c0, %c1]
    : memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf32, #sg_map_fp16_c>

  // CHECK: xegpu.load_nd %{{[0-9]}} : !xegpu.tensor_desc<8x16xf32, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>> -> vector<8x1xf32>
  %6 = xegpu.load_nd %5 : !xegpu.tensor_desc<8x16xf32, #sg_map_fp16_c> -> vector<8x1xf32>

  // CHECK: xegpu.create_nd_tdesc %{{arg[0-9]}}[%{{c[0-9]+}}, %{{c[0-9]+}}] : memref<24x32xf16>
  // CHECK-SAME: -> !xegpu.tensor_desc<8x16xf16, #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 2]>>
  %7 = xegpu.create_nd_tdesc %A[%c0, %c1]
      : memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16, #sg_map_fp16_d>
  // CHECK: xegpu.load_nd %{{[0-9]}} {vnni_axis = 1 : i64}
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf16, #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 2]>> -> vector<4x1x2xf16>
  %8 = xegpu.load_nd %7 {vnni_axis = 1} : !xegpu.tensor_desc<8x16xf16, #sg_map_fp16_d> -> vector<4x1x2xf16>

  return
}

#sg_map_bf16_a = #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 2]>
#sg_map_bf16_b = #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
#sg_map_bf16_c = #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
// CHECK-LABEL: func @test_load_nd_bf16({{.*}}) {
func.func @test_load_nd_bf16(%A: memref<24x32xbf16>, %B : memref<24x32xbf16>, %C : memref<24x32xbf16>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  // CHECK: xegpu.create_nd_tdesc %{{arg[0-9]}}[%{{c[0-9]+}}, %{{c[0-9]+}}] : memref<24x32xbf16>
  // CHECK-SAME: -> !xegpu.tensor_desc<8x16xbf16, #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 2]>>
  %1 = xegpu.create_nd_tdesc %A[%c0, %c1] : memref<24x32xbf16> -> !xegpu.tensor_desc<8x16xbf16, #sg_map_bf16_a>

  // CHECK: xegpu.load_nd %{{[0-9]}} {vnni_axis = 1 : i64}
  // CHECK-SAME: !xegpu.tensor_desc<8x16xbf16, #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 2]>> -> vector<4x1x2xbf16>
  %2 = xegpu.load_nd %1 {vnni_axis = 1} : !xegpu.tensor_desc<8x16xbf16, #sg_map_bf16_a> -> vector<4x1x2xbf16>

  // CHECK: xegpu.create_nd_tdesc %{{arg[0-9]}}[%{{c[0-9]+}}, %{{c[0-9]+}}] : memref<24x32xbf16>
  // CHECK-SAME: -> !xegpu.tensor_desc<16x16xbf16, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>
  %3 = xegpu.create_nd_tdesc %B[%c0, %c1] : memref<24x32xbf16> -> !xegpu.tensor_desc<16x16xbf16, #sg_map_bf16_b>

  // CHECK: xegpu.load_nd %{{[0-9]}} {vnni_axis = 0 : i64}
  // CHECK-SAME: !xegpu.tensor_desc<16x16xbf16, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>> -> vector<8x1x2xbf16>
  %4 = xegpu.load_nd %3 {vnni_axis = 0} : !xegpu.tensor_desc<16x16xbf16, #sg_map_bf16_b> -> vector<8x1x2xbf16>

  // CHECK: xegpu.create_nd_tdesc %{{arg[0-9]}}[%{{c[0-9]+}}, %{{c[0-9]+}}] : memref<24x32xbf16>
  // CHECK-SAME: -> !xegpu.tensor_desc<8x16xf32, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>
  %5 = xegpu.create_nd_tdesc %C[%c0, %c1] : memref<24x32xbf16> -> !xegpu.tensor_desc<8x16xf32, #sg_map_fp16_c>

  // CHECK: xegpu.load_nd %{{[0-9]}} : !xegpu.tensor_desc<8x16xf32, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>> -> vector<8x1xf32>
  %6 = xegpu.load_nd %5 : !xegpu.tensor_desc<8x16xf32, #sg_map_bf16_c> -> vector<8x1xf32>

  return
}

#sg_map_i8_a = #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 4]>
#sg_map_i8_b = #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
#sg_map_i8_c = #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
// CHECK-LABEL: func @test_load_nd_i8({{.*}}) {
func.func @test_load_nd_i8(%A: memref<64x64xi8>, %B : memref<64x64xi8>, %C : memref<64x64xi8>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  // CHECK: xegpu.create_nd_tdesc %{{arg[0-9]}}[%{{c[0-9]+}}, %{{c[0-9]+}}] : memref<64x64xi8>
  // CHECK-SAME: -> !xegpu.tensor_desc<8x32xi8, #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 4]>>
  %1 = xegpu.create_nd_tdesc %A[%c0, %c1] : memref<64x64xi8> -> !xegpu.tensor_desc<8x32xi8, #sg_map_i8_a>

  // CHECK: xegpu.load_nd %{{[0-9]}} {vnni_axis = 1 : i64}
  // CHECK-SAME: !xegpu.tensor_desc<8x32xi8, #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 4]>> -> vector<4x1x4xi8>
  %2 = xegpu.load_nd %1 {vnni_axis = 1} : !xegpu.tensor_desc<8x32xi8, #sg_map_i8_a> -> vector<4x1x4xi8>

  // CHECK: xegpu.create_nd_tdesc %{{arg[0-9]}}[%{{c[0-9]+}}, %{{c[0-9]+}}] : memref<64x64xi8>
  // CHECK-SAME: -> !xegpu.tensor_desc<32x16xi8, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>
  %3 = xegpu.create_nd_tdesc %B[%c0, %c1] : memref<64x64xi8> -> !xegpu.tensor_desc<32x16xi8, #sg_map_i8_b>

  // CHECK: xegpu.load_nd %{{[0-9]}} {vnni_axis = 0 : i64}
  // CHECK-SAME: !xegpu.tensor_desc<32x16xi8, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>> -> vector<8x1x4xi8>
  %4 = xegpu.load_nd %3 {vnni_axis = 0} : !xegpu.tensor_desc<32x16xi8, #sg_map_i8_b> -> vector<8x1x4xi8>

  // CHECK: xegpu.create_nd_tdesc %{{arg[0-9]}}[%{{c[0-9]+}}, %{{c[0-9]+}}] : memref<64x64xi8>
  // CHECK-SAME: -> !xegpu.tensor_desc<8x16xi32, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>
  %5 = xegpu.create_nd_tdesc %C[%c0, %c1] : memref<64x64xi8> -> !xegpu.tensor_desc<8x16xi32, #sg_map_i8_c>

  // CHECK: xegpu.load_nd %{{[0-9]}}
  // CHECK-SAME: !xegpu.tensor_desc<8x16xi32, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>> -> vector<8x1xi32>
  %6 = xegpu.load_nd %5 : !xegpu.tensor_desc<8x16xi32, #sg_map_i8_c> -> vector<8x1xi32>

  return
}

#sg_map_f64_a = #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 1]>
#sg_map_f64_b = #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 1]>
#sg_map_f64_c = #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 1]>
// CHECK-LABEL: func @test_load_nd_f64({{.*}}) {
func.func @test_load_nd_f64(%A: memref<64x64xf64>, %B : memref<64x64xf64>, %C : memref<64x64xf64>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: memref<64x64xf64>
  // CHECK-SAME: -> !xegpu.tensor_desc<4x8xf64, #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 1]>>
  %1 = xegpu.create_nd_tdesc %A[%c0, %c1]
      : memref<64x64xf64> -> !xegpu.tensor_desc<4x8xf64, #sg_map_f64_a>

  // CHECK: xegpu.load_nd
  // CHECK-SAME: !xegpu.tensor_desc<4x8xf64, #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 1]>>
  // CHECK-SAME: -> vector<2x1xf64>
  %2 = xegpu.load_nd %1 : !xegpu.tensor_desc<4x8xf64, #sg_map_f64_a> -> vector<2x1xf64>

  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME:  memref<64x64xf64>
  // CHECK-SAME:  -> !xegpu.tensor_desc<8x8xf64, #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 1]>>
  %3 = xegpu.create_nd_tdesc %B[%c0, %c1]
    : memref<64x64xf64> -> !xegpu.tensor_desc<8x8xf64, #sg_map_f64_b>

  // CHECK: xegpu.load_nd
  // CHECK-SAME: !xegpu.tensor_desc<8x8xf64, #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 1]>>
  // CHECK-SAME: -> vector<4x1xf64>
  %4 = xegpu.load_nd %3  : !xegpu.tensor_desc<8x8xf64, #sg_map_f64_b> -> vector<4x1xf64>

  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: memref<64x64xf64>
  // CHECK-SAME: -> !xegpu.tensor_desc<4x8xf64, #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 1]>>
  %5 = xegpu.create_nd_tdesc %C[%c0, %c1]
    : memref<64x64xf64> -> !xegpu.tensor_desc<4x8xf64, #sg_map_f64_c>

  // CHECK: xegpu.load_nd
  // CHECK-SAME: !xegpu.tensor_desc<4x8xf64, #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 1]>>
  // CHECK-SAME: -> vector<2x1xf64>
  %6 = xegpu.load_nd %5 : !xegpu.tensor_desc<4x8xf64, #sg_map_f64_c> -> vector<2x1xf64>

  return
}
