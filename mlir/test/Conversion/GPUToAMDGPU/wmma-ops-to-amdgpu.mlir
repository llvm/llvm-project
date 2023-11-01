// This file tests the conversion of GPU WMMA ops to ROCDL dialect.
// RUN: mlir-opt %s -convert-gpu-to-amdgpu='chipset=gfx1100 index-bitwidth=32' -split-input-file | FileCheck %s

gpu.module @test_module {
  // CHECK-LABEL: compute_op_f32_f16
  // CHECK-SAME: (%[[AOP:.*]]: !gpu.mma_matrix<16x16xf16, "AOp">, %[[BOP:.*]]: !gpu.mma_matrix<16x16xf16, "BOp">, %[[COP:.*]]: !gpu.mma_matrix<16x16xf32, "COp">)
  func.func @compute_op_f32_f16(%arg0: !gpu.mma_matrix<16x16xf16, "AOp">, %arg1: !gpu.mma_matrix<16x16xf16, "BOp">, %arg2: !gpu.mma_matrix<16x16xf32, "COp">) -> (!gpu.mma_matrix<16x16xf32, "COp">) {
    %0 = gpu.subgroup_mma_compute %arg0, %arg1, %arg2 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
    // CHECK:      %[[AOPMAT:.*]] = builtin.unrealized_conversion_cast %[[AOP]] : !gpu.mma_matrix<16x16xf16, "AOp"> to vector<16xf16>
    // CHECK-NEXT: %[[BOPMAT:.*]] = builtin.unrealized_conversion_cast %[[BOP]] : !gpu.mma_matrix<16x16xf16, "BOp"> to vector<16xf16>
    // CHECK-NEXT: %[[COPMAT:.*]] = builtin.unrealized_conversion_cast %[[COP]] : !gpu.mma_matrix<16x16xf32, "COp"> to vector<8xf32>
    // CHECK-NEXT: %[[OUTVEC:.*]] = amdgpu.wmma %[[AOPMAT]] * %[[BOPMAT]] + %[[COPMAT]] : vector<16xf16>, vector<16xf16>, vector<8xf32>
    // CHECK-NEXT: %[[OUTMAT:.*]] = builtin.unrealized_conversion_cast %[[OUTVEC]] : vector<8xf32> to !gpu.mma_matrix<16x16xf32, "COp">
    // CHECK-NEXT: return %[[OUTMAT]] : !gpu.mma_matrix<16x16xf32, "COp">
    return %0 : !gpu.mma_matrix<16x16xf32, "COp">
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: compute_op_f16_f16
  // CHECK-SAME: (%[[AOP:.*]]: !gpu.mma_matrix<16x16xf16, "AOp">, %[[BOP:.*]]: !gpu.mma_matrix<16x16xf16, "BOp">, %[[COP:.*]]: !gpu.mma_matrix<16x16xf16, "COp">)
  func.func @compute_op_f16_f16(%arg0: !gpu.mma_matrix<16x16xf16, "AOp">, %arg1: !gpu.mma_matrix<16x16xf16, "BOp">, %arg2: !gpu.mma_matrix<16x16xf16, "COp">) -> (!gpu.mma_matrix<16x16xf16, "COp">) {
    %0 = gpu.subgroup_mma_compute %arg0, %arg1, %arg2 {opSelect} : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
    // CHECK:      %[[AOPMAT:.*]] = builtin.unrealized_conversion_cast %[[AOP]] : !gpu.mma_matrix<16x16xf16, "AOp"> to vector<16xf16>
    // CHECK-NEXT: %[[BOPMAT:.*]] = builtin.unrealized_conversion_cast %[[BOP]] : !gpu.mma_matrix<16x16xf16, "BOp"> to vector<16xf16>
    // CHECK-NEXT: %[[COPMAT:.*]] = builtin.unrealized_conversion_cast %[[COP]] : !gpu.mma_matrix<16x16xf16, "COp"> to vector<16xf16>
    // CHECK-NEXT: %[[OUTVEC:.*]] = amdgpu.wmma %[[AOPMAT]] * %[[BOPMAT]] + %[[COPMAT]] {subwordOffset = 1 : i32} : vector<16xf16>, vector<16xf16>, vector<16xf16>
    // CHECK-NEXT: %[[OUTMAT:.*]] = builtin.unrealized_conversion_cast %[[OUTVEC]] : vector<16xf16> to !gpu.mma_matrix<16x16xf16, "COp">
    // CHECK-NEXT: return %[[OUTMAT]] : !gpu.mma_matrix<16x16xf16, "COp">
    return %0 : !gpu.mma_matrix<16x16xf16, "COp">
  }
}
