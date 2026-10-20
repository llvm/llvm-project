// RUN: mlir-opt %s -convert-gpu-to-nvvm -split-input-file | FileCheck %s

gpu.module @test_module_54 {
  // CHECK: llvm.func @__nv_fmaxf(f32, f32) -> f32
  // CHECK: llvm.func @__nv_fminf(f32, f32) -> f32
  // CHECK: llvm.func @__nv_fmax(f64, f64) -> f64
  // CHECK: llvm.func @__nv_fmin(f64, f64) -> f64
  // CHECK-LABEL: @gpu_fminmax
  func.func @gpu_fminmax(%arg1_f32: f32, %arg2_f32: f32, %arg1_f64: f64, %arg2_f64: f64) 
      -> (f32, f32, f64, f64) {
    // CHECK: llvm.call @__nv_fmaxf
    %max_f32 = arith.maxnumf %arg1_f32, %arg2_f32 : f32
    // CHECK: llvm.call @__nv_fminf
    %min_f32 = arith.minnumf %arg1_f32, %arg2_f32 : f32
    // CHECK: llvm.call @__nv_fmax(
    %max_f64 = arith.maxnumf %arg1_f64, %arg2_f64 : f64
    // CHECK: llvm.call @__nv_fmin(
    %min_f64 = arith.minnumf %arg1_f64, %arg2_f64 : f64
    return %max_f32, %min_f32, %max_f64, %min_f64 : f32, f32, f64, f64
  }
}
