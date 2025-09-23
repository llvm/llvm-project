// RUN: mlir-opt %s -convert-math-to-xevm | FileCheck %s

module @test_module {
  // CHECK: llvm.func @_Z22__spirv_ocl_native_expDh(f16) -> f16
  // CHECK: llvm.func @_Z22__spirv_ocl_native_expf(f32) -> f32
  // CHECK: llvm.func @_Z22__spirv_ocl_native_expd(f64) -> f64

  // CHECK: llvm.func @_Z22__spirv_ocl_native_expDv2_d(vector<2xf64>) -> vector<2xf64>
  // CHECK: llvm.func @_Z22__spirv_ocl_native_expDv3_d(vector<3xf64>) -> vector<3xf64>
  // CHECK: llvm.func @_Z22__spirv_ocl_native_expDv4_d(vector<4xf64>) -> vector<4xf64>
  // CHECK: llvm.func @_Z22__spirv_ocl_native_expDv8_d(vector<8xf64>) -> vector<8xf64>
  // CHECK: llvm.func @_Z22__spirv_ocl_native_expDv16_d(vector<16xf64>) -> vector<16xf64>
  // CHECK-LABEL: func @math_ops
  func.func @math_ops() {

    %c1_f16 = arith.constant 1. : f16
    %c1_f32 = arith.constant 1. : f32
    %c1_f64 = arith.constant 1. : f64

    // CHECK: math.exp
    %res_normal_f16 = math.exp %c1_f16 : f16
    // CHECK: math.exp
    %res_normal_f32 = math.exp %c1_f32 : f32
    // CHECK: math.exp
    %res_normal_f64 = math.exp %c1_f64 : f64

    // CHECK: llvm.call @_Z22__spirv_ocl_native_expDh(%{{.*}}) : (f16) -> f16
    %res_fast_f16 = math.exp %c1_f16 fastmath<fast> : f16
    // CHECK: llvm.call @_Z22__spirv_ocl_native_expf(%{{.*}}) : (f32) -> f32
    %res_fast_f32 = math.exp %c1_f32 fastmath<fast> : f32
    // CHECK: llvm.call @_Z22__spirv_ocl_native_expd(%{{.*}}) : (f64) -> f64
    %res_fast_f64 = math.exp %c1_f64 fastmath<fast> : f64
    
    // CHECK: llvm.call @_Z22__spirv_ocl_native_expDh(%{{.*}}) : (f16) -> f16
    %res_afn_f16 = math.exp %c1_f16 fastmath<afn> : f16
    // CHECK: llvm.call @_Z22__spirv_ocl_native_expf(%{{.*}}) : (f32) -> f32
    %res_afn_f32 = math.exp %c1_f32 fastmath<afn> : f32
    // CHECK: llvm.call @_Z22__spirv_ocl_native_expd(%{{.*}}) : (f64) -> f64
    %res_afn_f64 = math.exp %c1_f64 fastmath<afn> : f64

    // CHECK: math.exp
    %res_none_f16 = math.exp %c1_f16 fastmath<none> : f16
    // CHECK: math.exp
    %res_none_f32 = math.exp %c1_f32 fastmath<none> : f32
    // CHECK: math.exp
    %res_none_f64 = math.exp %c1_f64 fastmath<none> : f64

    %v2_c1_f64 = arith.constant dense<1.> : vector<2xf64>
    %v3_c1_f64 = arith.constant dense<1.> : vector<3xf64>
    %v4_c1_f64 = arith.constant dense<1.> : vector<4xf64>
    %v8_c1_f64 = arith.constant dense<1.> : vector<8xf64>
    %v16_c1_f64 = arith.constant dense<1.> : vector<16xf64>

    // CHECK: llvm.call @_Z22__spirv_ocl_native_expDv2_d(%{{.*}}) : (vector<2xf64>) -> vector<2xf64>
    %res_v2_f64 = math.exp %v2_c1_f64 fastmath<afn> : vector<2xf64>
    // CHECK: llvm.call @_Z22__spirv_ocl_native_expDv3_d(%{{.*}}) : (vector<3xf64>) -> vector<3xf64>
    %res_v3_f64 = math.exp %v3_c1_f64 fastmath<afn> : vector<3xf64>
    // CHECK: llvm.call @_Z22__spirv_ocl_native_expDv4_d(%{{.*}}) : (vector<4xf64>) -> vector<4xf64>
    %res_v4_f64 = math.exp %v4_c1_f64 fastmath<afn> : vector<4xf64>
    // CHECK: llvm.call @_Z22__spirv_ocl_native_expDv8_d(%{{.*}}) : (vector<8xf64>) -> vector<8xf64>
    %res_v8_f64 = math.exp %v8_c1_f64 fastmath<afn> : vector<8xf64>
    // CHECK: llvm.call @_Z22__spirv_ocl_native_expDv16_d(%{{.*}}) : (vector<16xf64>) -> vector<16xf64>
    %res_v16_f64 = math.exp %v16_c1_f64 fastmath<afn> : vector<16xf64>

    %v16_c1_f32 = arith.constant dense<1.> : vector<16xf32>
    %v4_c1_f16 = arith.constant dense<1.> : vector<4xf16>

    // CHECK: llvm.call @_Z22__spirv_ocl_native_expDv16_f(%{{.*}}) : (vector<16xf32>) -> vector<16xf32>
    %res_v16_f32 = math.exp %v16_c1_f32 fastmath<fast> : vector<16xf32>
    // CHECK: llvm.call @_Z22__spirv_ocl_native_expDv4_Dh(%{{.*}}) : (vector<4xf16>) -> vector<4xf16>
    %res_v4_f16 = math.exp %v4_c1_f16 fastmath<fast> : vector<4xf16>

    %v5_c1_f64 = arith.constant dense<1.> : vector<5xf64>
    %v32_c1_f64 = arith.constant dense<1.> : vector<32xf64>

    // CHECK: math.exp
    %res_v5_f64 = math.exp %v5_c1_f64 fastmath<afn> : vector<5xf64>
    // CHECK: math.exp
    %res_v32_f64 = math.exp %v32_c1_f64 fastmath<afn> : vector<32xf64>

    return
  }
}