// RUN: mlir-opt %s -convert-math-to-xevm \
// RUN:   | FileCheck %s -check-prefixes='CHECK,CHECK-ARITH' 
// RUN: mlir-opt %s -convert-math-to-xevm='convert-arith=false' \
// RUN:   | FileCheck %s -check-prefixes='CHECK,CHECK-NO-ARITH'

// RUN: mlir-opt --pass-pipeline="builtin.module(convert-math-to-xevm)" %s \
// RUN:   | FileCheck %s -check-prefixes='CHECK-MODULE,CHECK-ENTIRE-MODULE'
// RUN: mlir-opt --pass-pipeline="builtin.module(gpu.module(convert-math-to-xevm))" %s \
// RUN:   | FileCheck %s -check-prefixes='CHECK-MODULE,CHECK-ONLY-GPU'

// This test:
// - check that MathToXeVM converts fastmath math/arith ops properly;
// - check that MathToXeVM handles nested modules while respecting pass manager.

module @test_module {
  // CHECK-DAG: llvm.func @_Z22__spirv_ocl_native_expDh(f16) -> f16
  // CHECK-DAG: llvm.func @_Z22__spirv_ocl_native_expf(f32) -> f32
  // CHECK-DAG: llvm.func @_Z22__spirv_ocl_native_expd(f64) -> f64
  //
  // CHECK-DAG: llvm.func @_Z22__spirv_ocl_native_expDv2_d(vector<2xf64>) -> vector<2xf64>
  // CHECK-DAG: llvm.func @_Z22__spirv_ocl_native_expDv3_d(vector<3xf64>) -> vector<3xf64>
  // CHECK-DAG: llvm.func @_Z22__spirv_ocl_native_expDv4_d(vector<4xf64>) -> vector<4xf64>
  // CHECK-DAG: llvm.func @_Z22__spirv_ocl_native_expDv8_d(vector<8xf64>) -> vector<8xf64>
  // CHECK-DAG: llvm.func @_Z22__spirv_ocl_native_expDv16_d(vector<16xf64>) -> vector<16xf64>
  // CHECK-DAG: llvm.func @_Z22__spirv_ocl_native_expDv16_f(vector<16xf32>) -> vector<16xf32>
  // CHECK-DAG: llvm.func @_Z22__spirv_ocl_native_expDv4_Dh(vector<4xf16>) -> vector<4xf16>
  //
  // CHECK-DAG: llvm.func @_Z22__spirv_ocl_native_cosDh(f16) -> f16
  // CHECK-DAG: llvm.func @_Z23__spirv_ocl_native_exp2f(f32) -> f32
  // CHECK-DAG: llvm.func @_Z22__spirv_ocl_native_logDh(f16) -> f16
  // CHECK-DAG: llvm.func @_Z23__spirv_ocl_native_log2f(f32) -> f32
  // CHECK-DAG: llvm.func @_Z24__spirv_ocl_native_log10d(f64) -> f64
  // CHECK-DAG: llvm.func @_Z23__spirv_ocl_native_powrDhDh(f16, f16) -> f16
  // CHECK-DAG: llvm.func @_Z24__spirv_ocl_native_rsqrtd(f64) -> f64
  // CHECK-DAG: llvm.func @_Z22__spirv_ocl_native_sinDh(f16) -> f16
  // CHECK-DAG: llvm.func @_Z23__spirv_ocl_native_sqrtf(f32) -> f32
  // CHECK-DAG: llvm.func @_Z22__spirv_ocl_native_tand(f64) -> f64
  // CHECK-ARITH-DAG: llvm.func @_Z25__spirv_ocl_native_divideff(f32, f32) -> f32

  // CHECK-LABEL: func @math_ops
  func.func @math_ops() {

    %c1_f16 = arith.constant 1. : f16
    %c1_f32 = arith.constant 1. : f32
    %c1_f64 = arith.constant 1. : f64

    // CHECK: math.exp
    %exp_normal_f16 = math.exp %c1_f16 : f16
    // CHECK: math.exp
    %exp_normal_f32 = math.exp %c1_f32 : f32
    // CHECK: math.exp
    %exp_normal_f64 = math.exp %c1_f64 : f64

    // Check float operations are converted properly:

    // CHECK: llvm.call @_Z22__spirv_ocl_native_expDh(%{{.*}}) {fastmathFlags = #llvm.fastmath<fast>} : (f16) -> f16
    %exp_fast_f16 = math.exp %c1_f16 fastmath<fast> : f16
    // CHECK: llvm.call @_Z22__spirv_ocl_native_expf(%{{.*}}) {fastmathFlags = #llvm.fastmath<fast>} : (f32) -> f32
    %exp_fast_f32 = math.exp %c1_f32 fastmath<fast> : f32
    // CHECK: llvm.call @_Z22__spirv_ocl_native_expd(%{{.*}}) {fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
    %exp_fast_f64 = math.exp %c1_f64 fastmath<fast> : f64
    
    // CHECK: llvm.call @_Z22__spirv_ocl_native_expDh(%{{.*}}) {fastmathFlags = #llvm.fastmath<afn>} : (f16) -> f16
    %exp_afn_f16 = math.exp %c1_f16 fastmath<afn> : f16
    // CHECK: llvm.call @_Z22__spirv_ocl_native_expf(%{{.*}}) {fastmathFlags = #llvm.fastmath<afn>} : (f32) -> f32
    %exp_afn_f32 = math.exp %c1_f32 fastmath<afn> : f32
    // CHECK: llvm.call @_Z22__spirv_ocl_native_expd(%{{.*}}) {fastmathFlags = #llvm.fastmath<afn>} : (f64) -> f64
    %exp_afn_f64 = math.exp %c1_f64 fastmath<afn> : f64

    // CHECK: math.exp
    %exp_none_f16 = math.exp %c1_f16 fastmath<none> : f16
    // CHECK: math.exp
    %exp_none_f32 = math.exp %c1_f32 fastmath<none> : f32
    // CHECK: math.exp
    %exp_none_f64 = math.exp %c1_f64 fastmath<none> : f64

    // Check vector operations:

    %v2_c1_f64 = arith.constant dense<1.> : vector<2xf64>
    %v3_c1_f64 = arith.constant dense<1.> : vector<3xf64>
    %v4_c1_f64 = arith.constant dense<1.> : vector<4xf64>
    %v8_c1_f64 = arith.constant dense<1.> : vector<8xf64>
    %v16_c1_f64 = arith.constant dense<1.> : vector<16xf64>

    // CHECK: llvm.call @_Z22__spirv_ocl_native_expDv2_d(%{{.*}}) {fastmathFlags = #llvm.fastmath<afn>} : (vector<2xf64>) -> vector<2xf64>
    %exp_v2_f64 = math.exp %v2_c1_f64 fastmath<afn> : vector<2xf64>
    // CHECK: llvm.call @_Z22__spirv_ocl_native_expDv3_d(%{{.*}}) {fastmathFlags = #llvm.fastmath<afn>} : (vector<3xf64>) -> vector<3xf64>
    %exp_v3_f64 = math.exp %v3_c1_f64 fastmath<afn> : vector<3xf64>
    // CHECK: llvm.call @_Z22__spirv_ocl_native_expDv4_d(%{{.*}}) {fastmathFlags = #llvm.fastmath<afn>} : (vector<4xf64>) -> vector<4xf64>
    %exp_v4_f64 = math.exp %v4_c1_f64 fastmath<afn> : vector<4xf64>
    // CHECK: llvm.call @_Z22__spirv_ocl_native_expDv8_d(%{{.*}}) {fastmathFlags = #llvm.fastmath<afn>} : (vector<8xf64>) -> vector<8xf64>
    %exp_v8_f64 = math.exp %v8_c1_f64 fastmath<afn> : vector<8xf64>
    // CHECK: llvm.call @_Z22__spirv_ocl_native_expDv16_d(%{{.*}}) {fastmathFlags = #llvm.fastmath<afn>} : (vector<16xf64>) -> vector<16xf64>
    %exp_v16_f64 = math.exp %v16_c1_f64 fastmath<afn> : vector<16xf64>

    %v16_c1_f32 = arith.constant dense<1.> : vector<16xf32>
    %v4_c1_f16 = arith.constant dense<1.> : vector<4xf16>

    // CHECK: llvm.call @_Z22__spirv_ocl_native_expDv16_f(%{{.*}}) {fastmathFlags = #llvm.fastmath<fast>} : (vector<16xf32>) -> vector<16xf32>
    %exp_v16_f32 = math.exp %v16_c1_f32 fastmath<fast> : vector<16xf32>
    // CHECK: llvm.call @_Z22__spirv_ocl_native_expDv4_Dh(%{{.*}}) {fastmathFlags = #llvm.fastmath<fast>} : (vector<4xf16>) -> vector<4xf16>
    %exp_v4_f16 = math.exp %v4_c1_f16 fastmath<fast> : vector<4xf16>

    // Check unsupported vector sizes are not converted:

    %v5_c1_f64 = arith.constant dense<1.> : vector<5xf64>
    %v32_c1_f64 = arith.constant dense<1.> : vector<32xf64>

    // CHECK: math.exp
    %exp_v5_f64 = math.exp %v5_c1_f64 fastmath<afn> : vector<5xf64>
    // CHECK: math.exp
    %exp_v32_f64 = math.exp %v32_c1_f64 fastmath<afn> : vector<32xf64>

    // Check fastmath flags propagate properly:

    // CHECK: llvm.call @_Z22__spirv_ocl_native_expDh(%{{.*}}) {fastmathFlags = #llvm.fastmath<fast>} : (f16) -> f16
    %exp_fastmath_all_f16 = math.exp %c1_f16 fastmath<reassoc,nnan,ninf,nsz,arcp,contract,afn> : f16
    // CHECK: llvm.call @_Z22__spirv_ocl_native_expf(%{{.*}}) {fastmathFlags = #llvm.fastmath<nnan, ninf, nsz, arcp, contract, afn>} : (f32) -> f32
    %exp_fastmath_most_f32 = math.exp %c1_f32 fastmath<nnan,ninf,nsz,arcp,contract,afn> : f32
    // CHECK: llvm.call @_Z22__spirv_ocl_native_expf(%{{.*}}) {fastmathFlags = #llvm.fastmath<nnan, afn, reassoc>} : (f32) -> f32
    %exp_afn_reassoc_nnan_f32 = math.exp %c1_f32 fastmath<afn,reassoc,nnan> : f32

    // Check all other math operations:

    // CHECK: llvm.call @_Z22__spirv_ocl_native_cosDh(%{{.*}}) {fastmathFlags = #llvm.fastmath<afn>} : (f16) -> f16
    %cos_afn_f16 = math.cos %c1_f16 fastmath<afn> : f16

    // CHECK: llvm.call @_Z23__spirv_ocl_native_exp2f(%{{.*}}) {fastmathFlags = #llvm.fastmath<afn>} : (f32) -> f32
    %exp2_afn_f32 = math.exp2 %c1_f32 fastmath<afn> : f32

    // CHECK: llvm.call @_Z22__spirv_ocl_native_logDh(%{{.*}}) {fastmathFlags = #llvm.fastmath<afn>} : (f16) -> f16
    %log_afn_f16 = math.log %c1_f16 fastmath<afn> : f16

    // CHECK: llvm.call @_Z23__spirv_ocl_native_log2f(%{{.*}}) {fastmathFlags = #llvm.fastmath<afn>} : (f32) -> f32
    %log2_afn_f32 = math.log2 %c1_f32 fastmath<afn> : f32

    // CHECK: llvm.call @_Z24__spirv_ocl_native_log10d(%{{.*}}) {fastmathFlags = #llvm.fastmath<afn>} : (f64) -> f64
    %log10_afn_f64 = math.log10 %c1_f64 fastmath<afn> : f64

    // CHECK: llvm.call @_Z23__spirv_ocl_native_powrDhDh(%{{.*}}, %{{.*}}) {fastmathFlags = #llvm.fastmath<afn>} : (f16, f16) -> f16
    %powr_afn_f16 = math.powf %c1_f16, %c1_f16 fastmath<afn> : f16

    // CHECK: llvm.call @_Z24__spirv_ocl_native_rsqrtd(%{{.*}}) {fastmathFlags = #llvm.fastmath<afn>} : (f64) -> f64
    %rsqrt_afn_f64 = math.rsqrt %c1_f64 fastmath<afn> : f64

    // CHECK: llvm.call @_Z22__spirv_ocl_native_sinDh(%{{.*}}) {fastmathFlags = #llvm.fastmath<afn>} : (f16) -> f16
    %sin_afn_f16 = math.sin %c1_f16 fastmath<afn> : f16

    // CHECK: llvm.call @_Z23__spirv_ocl_native_sqrtf(%{{.*}}) {fastmathFlags = #llvm.fastmath<afn>} : (f32) -> f32
    %sqrt_afn_f32 = math.sqrt %c1_f32 fastmath<afn> : f32

    // CHECK: llvm.call @_Z22__spirv_ocl_native_tand(%{{.*}}) {fastmathFlags = #llvm.fastmath<afn>} : (f64) -> f64
    %tan_afn_f64 = math.tan %c1_f64 fastmath<afn> : f64

    %c6_9_f32 = arith.constant 6.9 : f32
    %c7_f32 = arith.constant 7. : f32

    // CHECK-ARITH: llvm.call @_Z25__spirv_ocl_native_divideff(%{{.*}}) {fastmathFlags = #llvm.fastmath<afn>} : (f32, f32) -> f32
    // CHECK-NO-ARITH: arith.divf
    %divf_afn_f32 = arith.divf %c6_9_f32, %c7_f32 fastmath<afn> : f32

    return
  }

  // Check that MathToXeVM handles nested modules while respecting pass manager:

  // CHECK-ENTIRE-MODULE: llvm.func @_Z22__spirv_ocl_native_expf(f32) -> f32
  // CHECK-ONLY-GPU-NOT: llvm.func @_Z22__spirv_ocl_native_expf(f32) -> f32
  
  // CHECK-MODULE-LABEL: @test_gpu
  gpu.module @test_gpu {
    // CHECK-MODULE: llvm.func @_Z22__spirv_ocl_native_expf(f32) -> f32
    gpu.func @exp_gpu() {
      %c1_f32 = arith.constant 1. : f32

      // CHECK-MODULE: math.exp
      %exp_normal_f32 = math.exp %c1_f32 : f32

      // CHECK-MODULE: llvm.call @_Z22__spirv_ocl_native_expf(%{{.*}}) {fastmathFlags = #llvm.fastmath<afn>} : (f32) -> f32
      %exp_fast_f32 = math.exp %c1_f32 fastmath<afn> : f32

      gpu.return
    }
  }

  // CHECK-MODULE-LABEL: @exp_func
  func.func @exp_func() {
    %c1_f32 = arith.constant 1. : f32

    // CHECK-MODULE: math.exp
    %exp_normal_f32 = math.exp %c1_f32 : f32

    // CHECK-ENTIRE-MODULE: llvm.call @_Z22__spirv_ocl_native_expf(%{{.*}}) {fastmathFlags = #llvm.fastmath<afn>} : (f32) -> f32
    // CHECK-ONLY-GPU: math.exp
    %exp_fast_f32 = math.exp %c1_f32 fastmath<afn> : f32

    return
  }
}
