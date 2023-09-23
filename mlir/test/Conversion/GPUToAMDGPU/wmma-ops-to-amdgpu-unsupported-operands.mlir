// This file tests the we error out properly when unsupported ops are
// encountered for GPU wmma ops to ROCDL conversion.

// RUN: mlir-opt %s -convert-gpu-to-amdgpu='chipset=gfx1100 index-bitwidth=32' -split-input-file -verify-diagnostics
gpu.module @test_module {
  // CHECK-LABEL: compute_op_f32_f16_transpose
  func.func @compute_op_f32_f16(%arg0: !gpu.mma_matrix<16x16xf16, "AOp">, %arg1: !gpu.mma_matrix<16x16xf16, "BOp">, %arg2: !gpu.mma_matrix<16x16xf32, "COp">) -> (!gpu.mma_matrix<16x16xf32, "COp">) {
    %0 = gpu.subgroup_mma_compute %arg0, %arg1, %arg2 {a_transpose}: !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
    // expected-error@-1 {{lowering with transpose is not supported. Please use transpose while loading/storing the operands.}}
    return %0 : !gpu.mma_matrix<16x16xf32, "COp">
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: compute_op_f32_f16_transpose
  func.func @compute_op_f32_f16(%arg0: !gpu.mma_matrix<16x16xf16, "AOp">, %arg1: !gpu.mma_matrix<16x16xf16, "BOp">, %arg2: !gpu.mma_matrix<16x16xf32, "COp">) -> (!gpu.mma_matrix<16x16xf32, "COp">) {
    %0 = gpu.subgroup_mma_compute %arg0, %arg1, %arg2 {b_transpose}: !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
    // expected-error@-1 {{lowering with transpose is not supported. Please use transpose while loading/storing the operands.}}
    return %0 : !gpu.mma_matrix<16x16xf32, "COp">
  }
}

// -----

gpu.module @test_module {
  func.func @compute_op_f32_f16(%arg0: !gpu.mma_matrix<16x8xf16, "AOp">, %arg1: !gpu.mma_matrix<8x16xf16, "BOp">, %arg2: !gpu.mma_matrix<16x16xf32, "COp">) -> (!gpu.mma_matrix<16x16xf32, "COp">) {
    %0 = gpu.subgroup_mma_compute %arg0, %arg1, %arg2 : !gpu.mma_matrix<16x8xf16, "AOp">, !gpu.mma_matrix<8x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
    // expected-error@-1 {{wmma ops of shape 16x16x16 are only supported.}}
    return %0 : !gpu.mma_matrix<16x16xf32, "COp">
  }
}
