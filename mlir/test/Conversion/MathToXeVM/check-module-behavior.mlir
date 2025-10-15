// RUN: mlir-opt --pass-pipeline="builtin.module(convert-math-to-xevm)" %s \
// RUN:   | FileCheck %s -check-prefixes='CHECK,CHECK-ENTIRE-MODULE'
// RUN: mlir-opt --pass-pipeline="builtin.module(gpu.module(convert-math-to-xevm))" %s \
// RUN:   | FileCheck %s -check-prefixes='CHECK,CHECK-ONLY-GPU'
//
// Check that MathToXeVM handles nested modules while respecting pass manager.

// CHECK-LABEL: @test_module
module @test_module {
  // CHECK-ENTIRE-MODULE: llvm.func @_Z22__spirv_ocl_native_expf(f32) -> f32
  // CHECK-ONLY-GPU-NOT: llvm.func @_Z22__spirv_ocl_native_expf(f32) -> f32

  // CHECK-LABEL: @test_gpu
  gpu.module @test_gpu {
    // CHECK: llvm.func @_Z22__spirv_ocl_native_expf(f32) -> f32
    gpu.func @exp_gpu() {
      %c1_f32 = arith.constant 1. : f32

      // CHECK: math.exp
      %exp_normal_f32 = math.exp %c1_f32 : f32

      // CHECK: llvm.call @_Z22__spirv_ocl_native_expf(%{{.*}}) {fastmathFlags = #llvm.fastmath<afn>} : (f32) -> f32
      %exp_fast_f32 = math.exp %c1_f32 fastmath<afn> : f32

      gpu.return
    }
  }

  // CHECK-LABEL: @exp_func
  func.func @exp_func() {
    %c1_f32 = arith.constant 1. : f32

    // CHECK: math.exp
    %exp_normal_f32 = math.exp %c1_f32 : f32

    // CHECK-ENTIRE-MODULE: llvm.call @_Z22__spirv_ocl_native_expf(%{{.*}}) {fastmathFlags = #llvm.fastmath<afn>} : (f32) -> f32
    // CHECK-ONLY-GPU: math.exp
    %exp_fast_f32 = math.exp %c1_f32 fastmath<afn> : f32

    return
  }
}
