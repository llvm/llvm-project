// RUN: mlir-opt %s --xevm-attach-target='module=xevm_* O=3 chip=pvc' -convert-vector-to-xegpu -split-input-file | FileCheck %s --check-prefixes=STORE-ND,CHECK
// RUN: mlir-opt %s -convert-vector-to-xegpu -split-input-file | FileCheck %s --check-prefixes=STORE-SCATTER,CHECK

gpu.module @xevm_module {
gpu.func @store_1D_vector(%vec: vector<8xf32>,
    %source: memref<8x16x32xf32>, %offset: index) {
  vector.transfer_write %vec, %source[%offset, %offset, %offset]
    {in_bounds = [true]}
    : vector<8xf32>, memref<8x16x32xf32>
  gpu.return
}

// CHECK-LABEL:  @store_1D_vector(
// CHECK-SAME:   %[[VEC:.+]]: vector<8xf32>,
// CHECK-SAME:   %[[SRC:.+]]: memref<8x16x32xf32>,
// CHECK-DAG:    %[[CST:.+]] = arith.constant dense<true> : vector<8xi1>
// CHECK-DAG:    %[[STEP:.+]] = vector.step
// CHECK    :    arith.muli {{.*}} : index
// CHECK    :    arith.addi {{.*}} : index
// CHECK-DAG:    %[[BCAST:.+]] = vector.broadcast {{.*}} : index to vector<8xindex>
// CHECK-DAG:    %[[IDX:.+]] = arith.addi %[[BCAST]], %{{.*}} : vector<8xindex>
// CHECK-DAG:    %[[COLLAPSE:.+]] = memref.extract_aligned_pointer_as_index %[[SRC]] : memref<8x16x32xf32> -> index
// CHECK-DAG:    %[[COLLAPSE_I:.+]] = arith.index_cast %[[COLLAPSE]] : index to i64
// CHECK:       xegpu.store %[[VEC]], %[[COLLAPSE_I]]{{\[}}%[[IDX]]{{\]}}, %[[CST]] : vector<8xf32>, i64, vector<8xindex>, vector<8xi1>
}

// -----
gpu.module @xevm_module {
gpu.func @store_2D_vector(%vec: vector<8x16xf32>,
    %source: memref<8x16x32xf32>, %offset: index) {
  vector.transfer_write %vec, %source[%offset, %offset, %offset]
    {in_bounds = [true, true]}
    : vector<8x16xf32>, memref<8x16x32xf32>
  gpu.return
}

// STORE-ND-LABEL: @store_2D_vector(
// STORE-ND-SAME:  %[[VEC:.+]]: vector<8x16xf32>,
// STORE-ND-SAME:  %[[SRC:.+]]: memref<8x16x32xf32>,
// STORE-ND-SAME:  %[[OFFSET:.+]]: index
// STORE-ND:       %[[COLLAPSED:.+]] = memref.subview %[[SRC]][%[[OFFSET]], 0, 0] [1, 16, 32] [1, 1, 1] : memref<8x16x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
// STORE-ND:       %[[DESC:.+]] = xegpu.create_nd_tdesc %[[COLLAPSED]] : memref<16x32xf32, strided<[32, 1], offset: ?>> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
// STORE-ND:       xegpu.store_nd %[[VEC]], %[[DESC]][%[[OFFSET]], %[[OFFSET]]] : vector<8x16xf32>

// STORE-SCATTER-LABEL:  @store_2D_vector(
// STORE-SCATTER-SAME:   %[[VEC:.+]]: vector<8x16xf32>,
// STORE-SCATTER-SAME:   %[[SRC:.+]]: memref<8x16x32xf32>,
// STORE-SCATTER-SAME:   %[[OFFSET:.+]]: index
// STORE-SCATTER:        %[[CST:.+]] = arith.constant dense<true> : vector<8x16xi1>
// STORE-SCATTER-COUNT2: %[[STEP:.+]] = vector.step
// STORE-SCATTER-COUNT2: vector.shape_cast {{.*}}
// STORE-SCATTER-COUNT2: vector.broadcast {{.*}} : vector<8x16xindex>
// STORE-SCATTER-DAG:    %[[BCAST2:.+]] = vector.broadcast {{.*}} : index to vector<8x16xindex>
// STORE-SCATTER-DAG:    %[[IDX:.+]] = arith.addi %[[BCAST2]], {{.*}} : vector<8x16xindex>
// STORE-SCATTER-DAG:    %[[COLLAPSE:.+]] = memref.extract_aligned_pointer_as_index %[[SRC]] : memref<8x16x32xf32> -> index
// STORE-SCATTER-DAG:    %[[COLLAPSE_I:.+]] = arith.index_cast %[[COLLAPSE]] : index to i64
// STORE-SCATTER:        xegpu.store %[[VEC]], %[[COLLAPSE_I]]{{\[}}%[[IDX]]{{\]}}, %[[CST]] : vector<8x16xf32>, i64, vector<8x16xindex>, vector<8x16xi1>
}

// -----
gpu.module @xevm_module {
gpu.func @store_dynamic_source(%vec: vector<8x16xf32>,
    %source: memref<?x?x?xf32>, %i: index, %j: index, %k: index) {
  vector.transfer_write %vec, %source[%i, %j, %k]
    {in_bounds = [true, true]}
    : vector<8x16xf32>, memref<?x?x?xf32>
  gpu.return
}

// STORE-ND-LABEL: @store_dynamic_source(
// STORE-ND-SAME:  %[[VEC:.+]]: vector<8x16xf32>,
// STORE-ND-SAME:  %[[SRC:.+]]: memref<?x?x?xf32>,
// STORE-ND-SAME:  %[[OFF0:.+]]: index, %[[OFF1:.+]]: index, %[[OFF2:.+]]: index
// STORE-ND:       %{{.*}}, %{{.*}}, %[[SIZES:.+]]:3, %{{.+}}:3 = memref.extract_strided_metadata %[[SRC]]
// STORE-ND:       %[[COLLAPSED:.+]] = memref.subview %[[SRC]][%[[OFF0]], 0, 0] [1, %[[SIZES]]#1, %[[SIZES]]#2] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
// STORE-ND:       %[[DESC:.+]] = xegpu.create_nd_tdesc %[[COLLAPSED]] : memref<?x?xf32, strided<[?, 1], offset: ?>> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
// STORE-ND:       xegpu.store_nd %[[VEC]], %[[DESC]][%[[OFF1]], %[[OFF2]]] : vector<8x16xf32>

// STORE-SCATTER-LABEL: @store_dynamic_source(
// STORE-SCATTER-SAME:  %[[VEC:.+]]: vector<8x16xf32>,
// STORE-SCATTER-SAME:  %[[SRC:.+]]: memref<?x?x?xf32>,
// STORE-SCATTER-DAG:   %[[CST:.+]] = arith.constant dense<true> : vector<8x16xi1>
// STORE-SCATTER-DAG:   memref.extract_strided_metadata %[[SRC]] : memref<?x?x?xf32> -> memref<f32>, index, index, index, index, index, index, index
// STORE-SCATTER-COUNT2: %[[STEP:.+]] = vector.step
// STORE-SCATTER-COUNT2: vector.shape_cast {{.*}}
// STORE-SCATTER-COUNT2: vector.broadcast {{.*}} : vector<8x16xindex>
// STORE-SCATTER-DAG:   %[[BCAST2:.+]] = vector.broadcast {{.*}} : index to vector<8x16xindex>
// STORE-SCATTER-DAG:   %[[IDX:.+]] = arith.addi %[[BCAST2]], {{.*}} : vector<8x16xindex>
// STORE-SCATTER-DAG:   %[[COLLAPSE:.+]] = memref.extract_aligned_pointer_as_index %[[SRC]] : memref<?x?x?xf32> -> index
// STORE-SCATTER-DAG:   %[[COLLAPSE_I:.+]] = arith.index_cast %[[COLLAPSE]] : index to i64
// STORE-SCATTER:       xegpu.store %[[VEC]], %[[COLLAPSE_I]]{{\[}}%[[IDX]]{{\]}}, %[[CST]] : vector<8x16xf32>, i64, vector<8x16xindex>, vector<8x16xi1>
}

// -----
// Equal vector and memref rank: the whole memref stays the create_nd source.
gpu.module @xevm_module {
gpu.func @store_high_dim_dyn(%vec: vector<1x1x8x16xf16>, %source: memref<?x?x8x16xf16>,
    %i: index, %j: index, %k: index, %l: index) {
  vector.transfer_write %vec, %source[%i, %j, %k, %l]
    {in_bounds = [true, true, true, true]}
    : vector<1x1x8x16xf16>, memref<?x?x8x16xf16>
  gpu.return
}

// STORE-ND-LABEL: @store_high_dim_dyn(
// STORE-ND-SAME:  %[[VEC:.+]]: vector<1x1x8x16xf16>, %[[SRC:.+]]: memref<?x?x8x16xf16>,
// STORE-ND-SAME:  %[[OFF0:.+]]: index, %[[OFF1:.+]]: index, %[[OFF2:.+]]: index, %[[OFF3:.+]]: index
// STORE-ND-NOT:   memref.subview
// STORE-ND:       %[[DESC:.+]] = xegpu.create_nd_tdesc %[[SRC]] : memref<?x?x8x16xf16> -> !xegpu.tensor_desc<1x1x8x16xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
// STORE-ND:       xegpu.store_nd %[[VEC]], %[[DESC]][%[[OFF0]], %[[OFF1]], %[[OFF2]], %[[OFF3]]] : vector<1x1x8x16xf16>

// STORE-SCATTER-LABEL: @store_high_dim_dyn(
// STORE-SCATTER-SAME:  %[[VEC:.+]]: vector<1x1x8x16xf16>, %[[SRC:.+]]: memref<?x?x8x16xf16>
// STORE-SCATTER:       %[[CST:.+]] = arith.constant dense<true> : vector<1x1x8x16xi1>
// STORE-SCATTER:       memref.extract_strided_metadata %[[SRC]]
// STORE-SCATTER:       %[[PTR:.+]] = memref.extract_aligned_pointer_as_index %[[SRC]] : memref<?x?x8x16xf16> -> index
// STORE-SCATTER:       %[[PTR_I:.+]] = arith.index_cast %[[PTR]] : index to i64
// STORE-SCATTER:       xegpu.store %[[VEC]], %[[PTR_I]]{{\[}}%{{.+}}{{\]}}, %[[CST]] : vector<1x1x8x16xf16>, i64, vector<1x1x8x16xindex>, vector<1x1x8x16xi1>
}

// -----
gpu.module @xevm_module {
gpu.func @store_out_of_bounds(%vec: vector<8x16xf32>,
    %source: memref<7x64xf32>, %offset: index) {
  vector.transfer_write %vec, %source[%offset, %offset]
    {in_bounds = [false, true]}
    : vector<8x16xf32>, memref<7x64xf32>
  gpu.return
}

// STORE-ND-LABEL:   @store_out_of_bounds(
// STORE-ND-SAME:  %[[VEC:.+]]: vector<8x16xf32>,
// STORE-ND-SAME:  %[[SRC:.+]]: memref<7x64xf32>,
// STORE-ND-SAME:  %[[OFFSET:.+]]: index
// STORE-ND:       %[[DESC:.+]] = xegpu.create_nd_tdesc
// STORE-ND-SAME:    %[[SRC]]
// STORE-ND-SAME:    memref<7x64xf32> -> !xegpu.tensor_desc<8x16xf32>
// STORE-ND:       xegpu.store_nd %[[VEC]], %[[DESC]][%[[OFFSET]], %[[OFFSET]]] : vector<8x16xf32>

// STORE-SCATTER-LABEL:  @store_out_of_bounds(
// STORE-SCATTER:   vector.transfer_write
}

// -----
gpu.module @xevm_module {
gpu.func @store_transposed(%vec: vector<8x16xf32>,
    %source: memref<32x64xf32>, %offset: index) {
  vector.transfer_write %vec, %source[%offset, %offset]
    {permutation_map = affine_map<(d0, d1) -> (d1, d0)>,
    in_bounds = [true, true]}
    : vector<8x16xf32>, memref<32x64xf32>
  gpu.return
}

// An nd block store cannot transpose, so a transposed write lowers through the
// scattered path identically on both the target and non-target runs.
// CHECK-LABEL:  @store_transposed(
// CHECK-SAME:   %[[VEC:.+]]: vector<8x16xf32>,
// CHECK-SAME:   %[[SRC:.+]]: memref<32x64xf32>,
// CHECK-SAME:   %[[OFFSET:.+]]: index
// CHECK:        %[[CST:.+]] = arith.constant dense<true> : vector<8x16xi1>
// CHECK     :   %[[STEP:.+]] = vector.step
// CHECK     :   vector.shape_cast {{.*}}
// CHECK     :   vector.broadcast {{.*}} : vector<8x16xindex>
// CHECK-DAG:    %[[BCAST2:.+]] = vector.broadcast {{.*}} : index to vector<8x16xindex>
// CHECK-DAG:    %[[IDX:.+]] = arith.addi %[[BCAST2]], {{.*}} : vector<8x16xindex>
// CHECK-DAG:    %[[COLLAPSE:.+]] = memref.extract_aligned_pointer_as_index %[[SRC]] : memref<32x64xf32> -> index
// CHECK-DAG:    %[[COLLAPSE_I:.+]] = arith.index_cast %[[COLLAPSE]] : index to i64
// CHECK:        xegpu.store %[[VEC]], %[[COLLAPSE_I]]{{\[}}%[[IDX]]{{\]}}, %[[CST]] : vector<8x16xf32>, i64, vector<8x16xindex>, vector<8x16xi1>
}

// -----
// A high-dim transposed write cannot use an nd block store (no transpose
// support), so it lowers through the scattered path on both runs.
gpu.module @xevm_module {
gpu.func @store_high_dim_transposed(%vec: vector<8x16x4xf32>,
    %source: memref<16x32x64xf32>, %offset: index) {
  vector.transfer_write %vec, %source[%offset, %offset, %offset]
    {permutation_map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>,
    in_bounds = [true, true, true]}
    : vector<8x16x4xf32>, memref<16x32x64xf32>
  gpu.return
}

// CHECK-LABEL:  @store_high_dim_transposed(
// CHECK:        xegpu.store {{.*}} : vector<8x16x4xf32>, i64, vector<8x16x4xindex>, vector<8x16x4xi1>
}

// -----
gpu.module @xevm_module {
gpu.func @store_high_dim_vector(%vec: vector<8x16x32xf32>,
    %source: memref<16x32x64xf32>, %offset: index) {
  vector.transfer_write %vec, %source[%offset, %offset, %offset]
    {in_bounds = [true, true, true]}
    : vector<8x16x32xf32>, memref<16x32x64xf32>
  gpu.return
}

// STORE-ND-LABEL:  @store_high_dim_vector(
// STORE-ND-SAME:   %[[VEC:.+]]: vector<8x16x32xf32>,
// STORE-ND-SAME:   %[[SRC:.+]]: memref<16x32x64xf32>
// STORE-ND:        %[[DESC:.+]] = xegpu.create_nd_tdesc %[[SRC]] : memref<16x32x64xf32>
// STORE-ND-SAME:     -> !xegpu.tensor_desc<8x16x32xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
// STORE-ND:        xegpu.store_nd %[[VEC]], %[[DESC]]
// STORE-ND-SAME:     : vector<8x16x32xf32>, !xegpu.tensor_desc<8x16x32xf32

// STORE-SCATTER-LABEL:  @store_high_dim_vector(
// STORE-SCATTER-SAME:   %[[VEC:.+]]: vector<8x16x32xf32>,
// STORE-SCATTER-SAME:   %[[SRC:.+]]: memref<16x32x64xf32>
// STORE-SCATTER:        %[[CST:.+]] = arith.constant dense<true> : vector<8x16x32xi1>
// STORE-SCATTER:        %[[CST_0:.+]] = arith.constant dense<64> : vector<16xindex>
// STORE-SCATTER:        %[[CST_1:.+]] = arith.constant dense<2048> : vector<8xindex>
// STORE-SCATTER:        %[[C2048:.+]] = arith.constant 2048 : index
// STORE-SCATTER:        %[[C64:.+]] = arith.constant 64 : index
// STORE-SCATTER-COUNT3: vector.step
// STORE-SCATTER-COUNT3: vector.shape_cast
// STORE-SCATTER-COUNT3: vector.broadcast {{.*}} : vector<8x16x32xindex>
// STORE-SCATTER-COUNT2: arith.addi {{.*}} : vector<8x16x32xindex>
// STORE-SCATTER:        %[[BCASTOFF:.+]] = vector.broadcast {{.*}} : index to vector<8x16x32xindex>
// STORE-SCATTER:        %[[IDX:.+]] = arith.addi %[[BCASTOFF]], {{.*}} : vector<8x16x32xindex>
// STORE-SCATTER:        %[[COLLAPSE:.+]] = memref.extract_aligned_pointer_as_index %[[SRC]] : memref<16x32x64xf32> -> index
// STORE-SCATTER:        %[[COLLAPSE_I:.+]] = arith.index_cast %[[COLLAPSE]] : index to i64
// STORE-SCATTER:        xegpu.store %[[VEC]], %[[COLLAPSE_I]][%[[IDX]]], %[[CST]] : vector<8x16x32xf32>, i64, vector<8x16x32xindex>, vector<8x16x32xi1>
}

// -----
gpu.module @xevm_module {
gpu.func @store_8D_vector(%vec: vector<2x2x2x2x2x2x2x2xf32>,
    %source: memref<2x2x2x2x2x2x2x2xf32>, %offset: index) {
  vector.transfer_write %vec, %source[%offset, %offset, %offset, %offset, %offset, %offset, %offset, %offset]
    {in_bounds = [true, true, true, true, true, true, true, true]}
    : vector<2x2x2x2x2x2x2x2xf32>, memref<2x2x2x2x2x2x2x2xf32>
  gpu.return
}

// STORE-ND-LABEL:  @store_8D_vector(
// STORE-ND-SAME:   %[[VEC:.+]]: vector<2x2x2x2x2x2x2x2xf32>,
// STORE-ND-SAME:   %[[SRC:.+]]: memref<2x2x2x2x2x2x2x2xf32>
// STORE-ND:        %[[DESC:.+]] = xegpu.create_nd_tdesc %[[SRC]] : memref<2x2x2x2x2x2x2x2xf32>
// STORE-ND-SAME:     -> !xegpu.tensor_desc<2x2x2x2x2x2x2x2xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
// STORE-ND:        xegpu.store_nd %[[VEC]], %[[DESC]]
// STORE-ND-SAME:     : vector<2x2x2x2x2x2x2x2xf32>, !xegpu.tensor_desc<2x2x2x2x2x2x2x2xf32

// STORE-SCATTER-LABEL:  @store_8D_vector(
// STORE-SCATTER-SAME:   %[[VEC:.+]]: vector<2x2x2x2x2x2x2x2xf32>,
// STORE-SCATTER-SAME:   %[[SRC:.+]]: memref<2x2x2x2x2x2x2x2xf32>
// STORE-SCATTER:        %[[CST:.+]] = arith.constant dense<true> : vector<2x2x2x2x2x2x2x2xi1>
// STORE-SCATTER-COUNT8: vector.step
// STORE-SCATTER-COUNT7: vector.shape_cast
// STORE-SCATTER-COUNT8: vector.broadcast {{.*}} : vector<2x2x2x2x2x2x2x2xindex>
// STORE-SCATTER-COUNT7: arith.addi {{.*}} : vector<2x2x2x2x2x2x2x2xindex>
// STORE-SCATTER:        %[[SPLAT:.+]] = vector.broadcast {{.*}} : index to vector<2x2x2x2x2x2x2x2xindex>
// STORE-SCATTER:        %[[IDX:.+]] = arith.addi %[[SPLAT]], {{.*}} : vector<2x2x2x2x2x2x2x2xindex>
// STORE-SCATTER:        %[[COLLAPSE:.+]] = memref.extract_aligned_pointer_as_index %[[SRC]] : memref<2x2x2x2x2x2x2x2xf32> -> index
// STORE-SCATTER:        %[[COLLAPSE_I:.+]] = arith.index_cast %[[COLLAPSE]] : index to i64
// STORE-SCATTER:        xegpu.store %[[VEC]], %[[COLLAPSE_I]][%[[IDX]]], %[[CST]] : vector<2x2x2x2x2x2x2x2xf32>, i64, vector<2x2x2x2x2x2x2x2xindex>, vector<2x2x2x2x2x2x2x2xi1>
}

// -----
gpu.module @xevm_module {
gpu.func @no_store_masked(%vec: vector<4xf32>,
    %source: memref<4xf32>, %offset: index) {
  %mask = arith.constant dense<[0, 1, 0, 1]> : vector<4xi1>
  vector.transfer_write %vec, %source[%offset], %mask
    {in_bounds = [true]}
    : vector<4xf32>, memref<4xf32>
  gpu.return
}

// CHECK-LABEL:  @no_store_masked(
// CHECK:        vector.transfer_write
}

// -----
gpu.module @xevm_module {
gpu.func @no_store_tensor(%vec: vector<8x16xf32>,
    %source: tensor<32x64xf32>, %offset: index) -> tensor<32x64xf32> {
  %0 = vector.transfer_write %vec, %source[%offset, %offset]
    {in_bounds = [true, true]}
    : vector<8x16xf32>, tensor<32x64xf32>
  gpu.return %0 : tensor<32x64xf32>
}

// CHECK-LABEL:  @no_store_tensor(
// CHECK:        vector.transfer_write
}

// -----
gpu.module @xevm_module {
gpu.func @no_store_non_unit_inner_stride(%vec: vector<8xf32>,
    %source: memref<32xf32, strided<[?], offset: ?>>, %offset: index) {
  vector.transfer_write %vec, %source[%offset]
    {in_bounds = [true]}
    : vector<8xf32>, memref<32xf32, strided<[?], offset: ?>>
  gpu.return
}

// CHECK-LABEL:  @no_store_non_unit_inner_stride(
// CHECK:        vector.transfer_write
}

// -----
gpu.module @xevm_module {
gpu.func @no_store_unsupported_map(%vec: vector<8x16xf32>,
    %source: memref<16x32x64xf32>, %offset: index) {
  vector.transfer_write %vec, %source[%offset, %offset, %offset]
    {permutation_map = affine_map<(d0, d1, d2) -> (d0, d2)>,
    in_bounds = [true, true]}
    : vector<8x16xf32>, memref<16x32x64xf32>
  gpu.return
}

// CHECK-LABEL:  @no_store_unsupported_map(
// CHECK:        vector.transfer_write
}

// -----
gpu.module @xevm_module {
gpu.func @no_store_out_of_bounds_1D_vector(%vec: vector<8xf32>,
    %source: memref<8x16x32xf32>, %offset: index) {
  vector.transfer_write %vec, %source[%offset, %offset, %offset]
    {in_bounds = [false]}
    : vector<8xf32>, memref<8x16x32xf32>
  gpu.return
}

// CHECK-LABEL:  @no_store_out_of_bounds_1D_vector(
// CHECK:        vector.transfer_write
}

// -----
gpu.module @xevm_module {
gpu.func @store_to_subview(%vec: vector<8xf16>,
    %source: memref<4096x4096xf16>, %off1: index, %off2: index) {
  %subview = memref.subview %source[%off1, %off2] [256, 256] [1, 1]
      : memref<4096x4096xf16>
        to memref<256x256xf16, strided<[4096, 1], offset: ?>>
  vector.transfer_write %vec, %subview[%off2, %off2]
      {in_bounds = [true]}
      : vector<8xf16>, memref<256x256xf16, strided<[4096, 1], offset: ?>>
  gpu.return
}

// CHECK-LABEL:  @store_to_subview(
// CHECK-SAME:   %[[VEC:.+]]: vector<8xf16>,
// CHECK-SAME:   %[[SRC:.+]]: memref<4096x4096xf16>,
// CHECK-SAME:   %[[OFF1:.+]]: index, %[[OFF2:.+]]: index
// CHECK:        %[[CST:.+]] = arith.constant dense<true> : vector<8xi1>
// CHECK:        %[[SUBVIEW:.+]] = memref.subview %[[SRC]][%[[OFF1]], %[[OFF2]]] [256, 256] [1, 1]
// CHECK-SAME:     : memref<4096x4096xf16> to memref<256x256xf16, strided<[4096, 1], offset: ?>>
// CHECK:        %[[BB:.+]], %[[OFFSET:.+]], {{.*}}, {{.*}} = memref.extract_strided_metadata %[[SUBVIEW]]
// CHECK-SAME:     : memref<256x256xf16, strided<[4096, 1], offset: ?>> -> memref<f16>, index, index, index, index, index
// CHECK:        %[[STEP:.+]] = vector.step : vector<8xindex>
// CHECK:        arith.muli {{.*}} : index
// CHECK:        arith.addi %[[OFFSET]]{{.*}} : index
// CHECK:        arith.addi {{.*}} : index
// CHECK:        %[[SPLAT:.+]] = vector.broadcast {{.*}} : index to vector<8xindex>
// CHECK:        %[[IDX:.+]] = arith.addi %[[SPLAT]], %[[STEP]] : vector<8xindex>
// CHECK:        %[[COLLAPSE:.+]] = memref.extract_aligned_pointer_as_index %[[SUBVIEW]]
// CHECK-SAME:     : memref<256x256xf16, strided<[4096, 1], offset: ?>> -> index
// CHECK:        %[[COLLAPSE_I:.+]] = arith.index_cast %[[COLLAPSE]] : index to i64
// CHECK:        xegpu.store %[[VEC]], %[[COLLAPSE_I]]{{\[}}%[[IDX]]{{\]}}, %[[CST]] : vector<8xf16>, i64, vector<8xindex>, vector<8xi1>
}

// -----
gpu.module @xevm_module {
gpu.func @store_2D_vector_addrspace3(%vec: vector<8x16xf32>,
    %source: memref<16x32xf32, 3>, %offset: index) {
  vector.transfer_write %vec, %source[%offset, %offset]
    {in_bounds = [true, true]}
    : vector<8x16xf32>, memref<16x32xf32, 3>
  gpu.return
}

// CHECK-LABEL: @store_2D_vector_addrspace3
// CHECK-SAME: %[[VEC:.+]]: vector<8x16xf32>
// CHECK-SAME: %[[SOURCE:.+]]: memref<16x32xf32, 3>
// CHECK-SAME: %[[OFFSET:.+]]: index
// CHECK: %[[MEM_DESC:.+]] = xegpu.create_mem_desc %[[SOURCE]] : memref<16x32xf32, 3> -> !xegpu.mem_desc<16x32xf32>
// CHECK: xegpu.store_matrix %[[VEC]], %[[MEM_DESC]][%[[OFFSET]], %[[OFFSET]]] : vector<8x16xf32>, !xegpu.mem_desc<16x32xf32>, index, index
// CHECK: gpu.return
}

// -----
gpu.module @xevm_module {
gpu.func @store_1D_vector_addrspace3(%vec: vector<8xf32>,
    %source: memref<32xf32, 3>, %offset: index) {
  vector.transfer_write %vec, %source[%offset]
    {in_bounds = [true]}
    : vector<8xf32>, memref<32xf32, 3>
  gpu.return
}

// CHECK-LABEL: @store_1D_vector_addrspace3
// CHECK-SAME: %[[VEC:.+]]: vector<8xf32>
// CHECK-SAME: %[[SOURCE:.+]]: memref<32xf32, 3>
// CHECK-SAME: %[[OFFSET:.+]]: index
// CHECK: %[[MEM_DESC:.+]] = xegpu.create_mem_desc %[[SOURCE]] : memref<32xf32, 3> -> !xegpu.mem_desc<32xf32>
// CHECK: xegpu.store_matrix %[[VEC]], %[[MEM_DESC]][%[[OFFSET]]] : vector<8xf32>, !xegpu.mem_desc<32xf32>, index
// CHECK: gpu.return
}

// -----
gpu.module @xevm_module {
gpu.func @store_0D_vector_unsupported(%vec: vector<f32>,
    %source: memref<3xf32>, %offset: index) {
  vector.transfer_write %vec, %source[%offset]
    : vector<f32>, memref<3xf32>
  gpu.return
}

// CHECK-LABEL: @store_0D_vector_unsupported
// CHECK: vector.transfer_write
}
