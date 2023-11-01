// RUN: mlir-opt %s -convert-gpu-to-rocdl='chipset=gfx900 index-bitwidth=32' -split-input-file -verify-diagnostics

gpu.module @main {
  // CHECK-LABEL: load_a_op_16_16_16_no_transpose_invalid_shape
  func.func @load_a_op_16_16_16_no_transpose()->(!gpu.mma_matrix<16x16xf16, "AOp">) {
    %wg = memref.alloca() {alignment = 32} : memref<32x32xf16, 3>
    %i = arith.constant 16 : index
    %j = arith.constant 16 : index
    %0 = gpu.subgroup_mma_load_matrix %wg[%i, %j] {leadDimension = 32 : index} : memref<32x32xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
    // expected-error@-1 {{wmma lowering is supported for gfx11 series only}}
    // expected-error@-2 {{failed to legalize operation 'gpu.subgroup_mma_load_matrix' that was explicitly marked illegal}}
    return %0 : !gpu.mma_matrix<16x16xf16, "AOp">
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: store_cop_f32
  func.func @store_cop_f32(%arg0: !gpu.mma_matrix<16x16xf32, "COp">) -> () {
    %wg_1 = memref.alloca() {alignment = 32} : memref<32x32xf32, 3>
    %i = arith.constant 16 : index
    %j = arith.constant 16 : index
    gpu.subgroup_mma_store_matrix %arg0, %wg_1[%i, %j] {leadDimension = 32 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<32x32xf32, 3>
    // expected-error@-1 {{wmma lowering is supported for gfx11 series only}}
    // expected-error@-2 {{failed to legalize operation 'gpu.subgroup_mma_store_matrix' that was explicitly marked illegal}}
    return
  }
}

