// This file tests the we error out properly when unsupported ops are
// encountered for GPU wmma ops to ROCDL conversion.
// RUN: mlir-opt %s -convert-gpu-to-rocdl='chipset=gfx1100 index-bitwidth=32' -split-input-file -verify-diagnostics

gpu.module @main {
  // CHECK-LABEL: load_a_op_16_16_16_no_transpose_invalid_shape
  func.func @load_a_op_16_16_16_no_transpose()->(!gpu.mma_matrix<32x8xf16, "AOp">) {
    %wg = memref.alloca() {alignment = 32} : memref<32x32xf16, 3>
    %i = arith.constant 16 : index
    %j = arith.constant 16 : index
    %0 = gpu.subgroup_mma_load_matrix %wg[%i, %j] {leadDimension = 32 : index} : memref<32x32xf16, 3> -> !gpu.mma_matrix<32x8xf16, "AOp">
    // expected-error@-1 {{wmma ops of shape 16x16x16 are only supported.}}
    // expected-error@-2 {{failed to legalize operation 'gpu.subgroup_mma_load_matrix' that was explicitly marked illegal}}
    return %0 : !gpu.mma_matrix<32x8xf16, "AOp">
  }
}

// -----

gpu.module @main {
  // CHECK-LABEL: load_a_op_16_16_16_transpose_invalid_shape
  func.func @load_a_op_16_16_16_transpose()->(!gpu.mma_matrix<32x8xf16, "AOp">) {
    %wg = memref.alloca() {alignment = 32} : memref<32x32xf16, 3>
    %i = arith.constant 16 : index
    %j = arith.constant 16 : index
    %0 = gpu.subgroup_mma_load_matrix %wg[%i, %j] {leadDimension = 32 : index, transpose} : memref<32x32xf16, 3> -> !gpu.mma_matrix<32x8xf16, "AOp">
    // expected-error@-1 {{wmma ops of shape 16x16x16 are only supported.}}
    // expected-error@-2 {{failed to legalize operation 'gpu.subgroup_mma_load_matrix' that was explicitly marked illegal}}
    return %0 : !gpu.mma_matrix<32x8xf16, "AOp">
  }
}

// -----

gpu.module @main {
  // CHECK-LABEL: load_a_op_16_16_16_no_transpose_invalid_types
  func.func @load_a_op_16_16_16_no_transpose_invalid_types()->(!gpu.mma_matrix<16x16xf16, "AOp">) {
    %wg = memref.alloca() {alignment = 32} : memref<32x32xf32, 3>
    %i = arith.constant 16 : index
    %j = arith.constant 16 : index
    %0 = gpu.subgroup_mma_load_matrix %wg[%i, %j] {leadDimension = 32 : index} : memref<32x32xf32, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
    // expected-error@-1 {{src memref type and mma matrix element type must be same}}
    // expected-error@-2 {{failed to legalize operation 'gpu.subgroup_mma_load_matrix' that was explicitly marked illegal}}
    return %0 : !gpu.mma_matrix<16x16xf16, "AOp">
  }
}

// -----

gpu.module @main {
  // CHECK-LABEL: load_b_op_16_16_16_no_transpose_invalid_shape
  func.func @load_b_op_16_16_16_no_transpose()->(!gpu.mma_matrix<32x8xf16, "BOp">) {
    %wg = memref.alloca() {alignment = 32} : memref<32x32xf16, 3>
    %i = arith.constant 16 : index
    %j = arith.constant 16 : index
    %0 = gpu.subgroup_mma_load_matrix %wg[%i, %j] {leadDimension = 32 : index} : memref<32x32xf16, 3> -> !gpu.mma_matrix<32x8xf16, "BOp">
    // expected-error@-1 {{wmma ops of shape 16x16x16 are only supported.}}
    // expected-error@-2 {{failed to legalize operation 'gpu.subgroup_mma_load_matrix' that was explicitly marked illegal}}
    return %0 : !gpu.mma_matrix<32x8xf16, "BOp">
  }
}

// -----

gpu.module @main {
  // CHECK-LABEL: load_b_op_16_16_16_transpose_invalid_shape
  func.func @load_b_op_16_16_16_transpose()->(!gpu.mma_matrix<32x8xf16, "BOp">) {
    %wg = memref.alloca() {alignment = 32} : memref<32x32xf16, 3>
    %i = arith.constant 16 : index
    %j = arith.constant 16 : index
    %0 = gpu.subgroup_mma_load_matrix %wg[%i, %j] {leadDimension = 32 : index, transpose} : memref<32x32xf16, 3> -> !gpu.mma_matrix<32x8xf16, "BOp">
    // expected-error@-1 {{wmma ops of shape 16x16x16 are only supported.}}
    // expected-error@-2 {{failed to legalize operation 'gpu.subgroup_mma_load_matrix' that was explicitly marked illegal}}
    return %0 : !gpu.mma_matrix<32x8xf16, "BOp">
  }
}

// -----

gpu.module @main {
  // CHECK-LABEL: load_b_op_16_16_16_no_transpose_invalid_types
  func.func @load_b_op_16_16_16_no_transpose_invalid_types()->(!gpu.mma_matrix<16x16xf16, "BOp">) {
    %wg = memref.alloca() {alignment = 32} : memref<32x32xf32, 3>
    %i = arith.constant 16 : index
    %j = arith.constant 16 : index
    %0 = gpu.subgroup_mma_load_matrix %wg[%i, %j] {leadDimension = 32 : index} : memref<32x32xf32, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
    // expected-error@-1 {{src memref type and mma matrix element type must be same}}
    // expected-error@-2 {{failed to legalize operation 'gpu.subgroup_mma_load_matrix' that was explicitly marked illegal}}
    return %0 : !gpu.mma_matrix<16x16xf16, "BOp">
  }
}

// -----

gpu.module @main {
  // CHECK-LABEL: load_c_op_16_16_16_no_transpose_invalid_shape
  func.func @load_c_op_16_16_16_no_transpose()->(!gpu.mma_matrix<32x8xf16, "COp">) {
    %wg = memref.alloca() {alignment = 32} : memref<32x32xf16, 3>
    %i = arith.constant 16 : index
    %j = arith.constant 16 : index
    %0 = gpu.subgroup_mma_load_matrix %wg[%i, %j] {leadDimension = 32 : index} : memref<32x32xf16, 3> -> !gpu.mma_matrix<32x8xf16, "COp">
    // expected-error@-1 {{wmma ops of shape 16x16x16 are only supported.}}
    // expected-error@-2 {{failed to legalize operation 'gpu.subgroup_mma_load_matrix' that was explicitly marked illegal}}
    return %0 : !gpu.mma_matrix<32x8xf16, "COp">
  }
}

// -----

gpu.module @main {
  // CHECK-LABEL: load_c_op_16_16_16_transpose_invalid_shape
  func.func @load_c_op_16_16_16_transpose()->(!gpu.mma_matrix<32x8xf16, "COp">) {
    %wg = memref.alloca() {alignment = 32} : memref<32x32xf16, 3>
    %i = arith.constant 16 : index
    %j = arith.constant 16 : index
    %0 = gpu.subgroup_mma_load_matrix %wg[%i, %j] {leadDimension = 32 : index, transpose} : memref<32x32xf16, 3> -> !gpu.mma_matrix<32x8xf16, "COp">
    // expected-error@-1 {{wmma ops of shape 16x16x16 are only supported.}}
    // expected-error@-2 {{failed to legalize operation 'gpu.subgroup_mma_load_matrix' that was explicitly marked illegal}}
    return %0 : !gpu.mma_matrix<32x8xf16, "COp">
  }
}

// -----

gpu.module @main {
  // CHECK-LABEL: load_c_op_16_16_16_no_transpose_invalid_types
  func.func @load_c_op_16_16_16_no_transpose_invalid_types()->(!gpu.mma_matrix<16x16xf16, "COp">) {
    %wg = memref.alloca() {alignment = 32} : memref<32x32xf32, 3>
    %i = arith.constant 16 : index
    %j = arith.constant 16 : index
    %0 = gpu.subgroup_mma_load_matrix %wg[%i, %j] {leadDimension = 32 : index} : memref<32x32xf32, 3> -> !gpu.mma_matrix<16x16xf16, "COp">
    // expected-error@-1 {{src memref type and mma matrix element type must be same}}
    // expected-error@-2 {{failed to legalize operation 'gpu.subgroup_mma_load_matrix' that was explicitly marked illegal}}
    return %0 : !gpu.mma_matrix<16x16xf16, "COp">
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: store_cop_f32
  func.func @store_cop_f32(%arg0: !gpu.mma_matrix<32x8xf32, "COp">) -> () {
    %wg_1 = memref.alloca() {alignment = 32} : memref<32x32xf32, 3>
    %i = arith.constant 16 : index
    %j = arith.constant 16 : index
    gpu.subgroup_mma_store_matrix %arg0, %wg_1[%i, %j] {leadDimension = 32 : index} : !gpu.mma_matrix<32x8xf32, "COp">, memref<32x32xf32, 3>
    // expected-error@-1 {{wmma ops of shape 16x16x16 are only supported.}}
    // expected-error@-2 {{failed to legalize operation 'gpu.subgroup_mma_store_matrix' that was explicitly marked illegal}}
    return
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: store_cop_f32
  func.func @store_cop_f32(%arg0: !gpu.mma_matrix<16x16xf32, "COp">) -> () {
    %wg_1 = memref.alloca() {alignment = 32} : memref<32x32xf32, 3>
    %i = arith.constant 16 : index
    %j = arith.constant 16 : index
    gpu.subgroup_mma_store_matrix %arg0, %wg_1[%i, %j] {leadDimension = 32 : index, transpose} : !gpu.mma_matrix<16x16xf32, "COp">, memref<32x32xf32, 3>
    // expected-error@-1 {{lowering with transpose is not supported.}}
    // expected-error@-2 {{failed to legalize operation 'gpu.subgroup_mma_store_matrix' that was explicitly marked illegal}}
    return
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: store_cop_f32
  func.func @store_cop_f32(%arg0: !gpu.mma_matrix<16x16xf32, "COp">) -> () {
    %wg_1 = memref.alloca() {alignment = 32} : memref<32x32xf16, 3>
    %i = arith.constant 16 : index
    %j = arith.constant 16 : index
    gpu.subgroup_mma_store_matrix %arg0, %wg_1[%i, %j] {leadDimension = 32 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<32x32xf16, 3>
    // expected-error@-1 {{dst memref type and mma matrix element type must be same}}
    // expected-error@-2 {{failed to legalize operation 'gpu.subgroup_mma_store_matrix' that was explicitly marked illegal}}
    return
  }
}
