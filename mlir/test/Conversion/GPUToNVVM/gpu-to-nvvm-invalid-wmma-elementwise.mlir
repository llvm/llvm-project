// RUN: mlir-opt %s -convert-gpu-to-nvvm -split-input-file -verify-diagnostics

// The PTX ISA does not define a conversion between f16 and f32 accumulator
// fragments.

gpu.module @test_module {
  func.func @wmma_elementwise_extf(%A : !gpu.mma_matrix<16x16xf16, "COp">) -> !gpu.mma_matrix<16x16xf32, "COp"> {
    // expected-error @+1 {{failed to legalize operation 'gpu.subgroup_mma_elementwise' that was explicitly marked illegal}}
    %0 = gpu.subgroup_mma_elementwise extf %A : (!gpu.mma_matrix<16x16xf16, "COp">) -> !gpu.mma_matrix<16x16xf32, "COp">
    return %0 : !gpu.mma_matrix<16x16xf32, "COp">
  }
}

// -----

gpu.module @test_module {
  func.func @wmma_elementwise_truncf(%A : !gpu.mma_matrix<16x16xf32, "COp">) -> !gpu.mma_matrix<16x16xf16, "COp"> {
    // expected-error @+1 {{failed to legalize operation 'gpu.subgroup_mma_elementwise' that was explicitly marked illegal}}
    %0 = gpu.subgroup_mma_elementwise truncf %A : (!gpu.mma_matrix<16x16xf32, "COp">) -> !gpu.mma_matrix<16x16xf16, "COp">
    return %0 : !gpu.mma_matrix<16x16xf16, "COp">
  }
}

// -----

// s8 multiplicand fragments pack four elements into each i32 register.

gpu.module @test_module {
  func.func @wmma_elementwise_packed_s8(%A : !gpu.mma_matrix<16x16xsi8, "AOp">) -> !gpu.mma_matrix<16x16xsi8, "AOp"> {
    // expected-error @+1 {{failed to legalize operation 'gpu.subgroup_mma_elementwise' that was explicitly marked illegal}}
    %0 = gpu.subgroup_mma_elementwise addi %A, %A : (!gpu.mma_matrix<16x16xsi8, "AOp">, !gpu.mma_matrix<16x16xsi8, "AOp">) -> !gpu.mma_matrix<16x16xsi8, "AOp">
    return %0 : !gpu.mma_matrix<16x16xsi8, "AOp">
  }
}

// -----

// f32 multiplicand fragments hold tf32 bit patterns in i32 registers.

gpu.module @test_module {
  func.func @wmma_elementwise_packed_tf32(%A : !gpu.mma_matrix<16x8xf32, "AOp">) -> !gpu.mma_matrix<16x8xf32, "AOp"> {
    // expected-error @+1 {{failed to legalize operation 'gpu.subgroup_mma_elementwise' that was explicitly marked illegal}}
    %0 = gpu.subgroup_mma_elementwise addf %A, %A : (!gpu.mma_matrix<16x8xf32, "AOp">, !gpu.mma_matrix<16x8xf32, "AOp">) -> !gpu.mma_matrix<16x8xf32, "AOp">
    return %0 : !gpu.mma_matrix<16x8xf32, "AOp">
  }
}
