// RUN: mlir-opt %s -convert-gpu-to-amdgpu='chipset=gfx900 index-bitwidth=32' -split-input-file -verify-diagnostics

gpu.module @test_module {
  func.func @compute_op_f32_f16(%arg0: !gpu.mma_matrix<16x16xf16, "AOp">, %arg1: !gpu.mma_matrix<16x16xf16, "BOp">, %arg2: !gpu.mma_matrix<16x16xf32, "COp">) -> (!gpu.mma_matrix<16x16xf32, "COp">) {
    %0 = gpu.subgroup_mma_compute %arg0, %arg1, %arg2 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
    // expected-error@-1 {{wmma lowering is supported for gfx11 series only}}
    return %0 : !gpu.mma_matrix<16x16xf32, "COp">
  }
}

