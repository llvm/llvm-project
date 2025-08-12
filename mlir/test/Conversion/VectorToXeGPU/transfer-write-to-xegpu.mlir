// RUN: mlir-opt %s --xevm-attach-target='module=xevm.* O=3 chip=pvc' -convert-vector-to-xegpu -split-input-file | FileCheck %s --check-prefix=STORE-ND
// RUN: mlir-opt %s -convert-vector-to-xegpu -split-input-file | FileCheck %s --check-prefix=STORE-SCATTER


gpu.module @xevm_module {
gpu.func @store_1D_vector(%vec: vector<8xf32>,
    %source: memref<8x16x32xf32>, %offset: index) {
  vector.transfer_write %vec, %source[%offset, %offset, %offset]
    {in_bounds = [true]}
    : vector<8xf32>, memref<8x16x32xf32>
  gpu.return
}

// STORE-ND-LABEL: @store_1D_vector(
// STORE-ND-SAME:  %[[VEC:.+]]: vector<8xf32>,
// STORE-ND-SAME:  %[[SRC:.+]]: memref<8x16x32xf32>,
// STORE-ND-SAME:  %[[OFFSET:.+]]: index
// STORE-ND:       %[[DESC:.+]] = xegpu.create_nd_tdesc
// STORE-ND-SAME:    %[[SRC]][%[[OFFSET]], %[[OFFSET]], %[[OFFSET]]]
// STORE-ND-SAME:    memref<8x16x32xf32> -> !xegpu.tensor_desc<8xf32,
// STORE-ND-SAME:    boundary_check = false
// STORE-ND:       xegpu.store_nd %[[VEC]], %[[DESC]] : vector<8xf32>

// STORE-SCATTER-LABEL:  @store_1D_vector(
// STORE-SCATTER-SAME:   %[[VEC:.+]]: vector<8xf32>,
// STORE-SCATTER-SAME:   %[[SRC:.+]]: memref<8x16x32xf32>,
// STORE-SCATTER-DAG:        %[[CST:.+]] = arith.constant dense<true> : vector<8xi1>
// STORE-SCATTER-DAG:        %[[STEP:.+]] = vector.step
// STORE-SCATTER-COUNT2: arith.muli {{.*}} : index
// STORE-SCATTER-COUNT2: arith.addi {{.*}} : index
// STORE-SCATTER-DAG:    %[[BCAST:.+]] = vector.broadcast {{.*}} : index to vector<8xindex>
// STORE-SCATTER-DAG:    %[[IDX:.+]] = arith.addi %[[BCAST]], %{{.*}} : vector<8xindex>
// STORE-SCATTER-DAG:    %[[COLLAPSE:.+]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1, 2]{{\]}} : memref<8x16x32xf32> into memref<4096xf32>
// STORE-SCATTER:       xegpu.store %[[VEC]], %[[COLLAPSE]]{{\[}}%[[IDX]]{{\]}}, %[[CST]] : vector<8xf32>, memref<4096xf32>, vector<8xindex>, vector<8xi1>
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
// STORE-ND:       %[[DESC:.+]] = xegpu.create_nd_tdesc
// STORE-ND-SAME:    %[[SRC]][%[[OFFSET]], %[[OFFSET]], %[[OFFSET]]]
// STORE-ND-SAME:    memref<8x16x32xf32> -> !xegpu.tensor_desc<8x16xf32,
// STORE-ND-SAME:    boundary_check = false
// STORE-ND:       xegpu.store_nd %[[VEC]], %[[DESC]] : vector<8x16xf32>

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
// STORE-SCATTER-DAG:    %[[COLLAPSE:.+]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1, 2]{{\]}} : memref<8x16x32xf32> into memref<4096xf32>
// STORE-SCATTER:        xegpu.store %[[VEC]], %[[COLLAPSE]]{{\[}}%[[IDX]]{{\]}}, %[[CST]] : vector<8x16xf32>, memref<4096xf32>, vector<8x16xindex>, vector<8x16xi1>
}

// -----
gpu.module @xevm_module {
gpu.func @store_dynamic_source(%vec: vector<8x16xf32>,
    %source: memref<?x?x?xf32>, %offset: index) {
  vector.transfer_write %vec, %source[%offset, %offset, %offset]
    {in_bounds = [true, true]}
    : vector<8x16xf32>, memref<?x?x?xf32>
  gpu.return
}

// STORE-ND-LABEL: @store_dynamic_source(
// STORE-ND-SAME:  %[[VEC:.+]]: vector<8x16xf32>,
// STORE-ND-SAME:  %[[SRC:.+]]: memref<?x?x?xf32>,
// STORE-ND-SAME:  %[[OFFSET:.+]]: index
// STORE-ND-DAG:   %[[C0:.+]] = arith.constant 0 : index
// STORE-ND-DAG:   %[[C1:.+]] = arith.constant 1 : index
// STORE-ND-DAG:   %[[C2:.+]] = arith.constant 2 : index
// STORE-ND-DAG:   %[[DIM_0:.+]] = memref.dim %[[SRC]], %[[C0]]
// STORE-ND-DAG:   %[[DIM_1:.+]] = memref.dim %[[SRC]], %[[C1]]
// STORE-ND-DAG:   %[[DIM_2:.+]] = memref.dim %[[SRC]], %[[C2]]
// STORE-ND:       %[[DIM_0_STRIDE:.+]] = arith.muli %[[DIM_2]], %[[DIM_1]]
// STORE-ND:       %[[DESC:.+]] = xegpu.create_nd_tdesc %[[SRC]][%[[OFFSET]], %[[OFFSET]], %[[OFFSET]]]
// STORE-ND-SAME:  , shape : [%[[DIM_0]], %[[DIM_1]], %[[DIM_2]]], strides : [%[[DIM_0_STRIDE]], %[[DIM_2]], 1]
// STORE-ND-SAME:    memref<?x?x?xf32> -> !xegpu.tensor_desc<8x16xf32
// STORE-ND:       xegpu.store_nd %[[VEC]], %[[DESC]] : vector<8x16xf32>

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
// STORE-SCATTER-DAG:   %[[COLLAPSE:.+]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1, 2]{{\]}} : memref<?x?x?xf32> into memref<?xf32>
// STORE-SCATTER:       xegpu.store %[[VEC]], %[[COLLAPSE]]{{\[}}%[[IDX]]{{\]}}, %[[CST]] : vector<8x16xf32>, memref<?xf32>, vector<8x16xindex>, vector<8x16xi1>
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
// STORE-ND-SAME:    %[[SRC]][%[[OFFSET]], %[[OFFSET]]]
// STORE-ND-SAME:    memref<7x64xf32> -> !xegpu.tensor_desc<8x16xf32>
// STORE-ND:       xegpu.store_nd %[[VEC]], %[[DESC]] : vector<8x16xf32>

// STORE-SCATTER-LABEL:  @store_out_of_bounds(
// STORE-SCATTER:   vector.transfer_write
}

// -----
gpu.module @xevm_module {
gpu.func @no_store_transposed(%vec: vector<8x16xf32>,
    %source: memref<32x64xf32>, %offset: index) {
  vector.transfer_write %vec, %source[%offset, %offset]
    {permutation_map = affine_map<(d0, d1) -> (d1, d0)>,
    in_bounds = [true, true]}
    : vector<8x16xf32>, memref<32x64xf32>
  gpu.return
}

// STORE-ND-LABEL: @no_store_transposed(
// STORE-ND:       vector.transfer_write

// STORE-SCATTER-LABEL:  @no_store_transposed(
// STORE-SCATTER-SAME:   %[[VEC:.+]]: vector<8x16xf32>,
// STORE-SCATTER-SAME:   %[[SRC:.+]]: memref<32x64xf32>,
// STORE-SCATTER-SAME:   %[[OFFSET:.+]]: index
// STORE-SCATTER:        %[[CST:.+]] = arith.constant dense<true> : vector<8x16xi1>
// STORE-SCATTER-COUNT2: %[[STEP:.+]] = vector.step
// STORE-SCATTER-COUNT2: vector.shape_cast {{.*}}
// STORE-SCATTER-COUNT2: vector.broadcast {{.*}} : vector<8x16xindex>
// STORE-SCATTER-DAG:    %[[BCAST2:.+]] = vector.broadcast {{.*}} : index to vector<8x16xindex>
// STORE-SCATTER-DAG:    %[[IDX:.+]] = arith.addi %[[BCAST2]], {{.*}} : vector<8x16xindex>
// STORE-SCATTER-DAG:    %[[COLLAPSE:.+]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1]{{\]}} : memref<32x64xf32> into memref<2048xf32>
// STORE-SCATTER:        xegpu.store %[[VEC]], %[[COLLAPSE]]{{\[}}%[[IDX]]{{\]}}, %[[CST]] : vector<8x16xf32>, memref<2048xf32>, vector<8x16xindex>, vector<8x16xi1>
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

// STORE-ND-LABEL: @store_high_dim_vector(
// STORE-ND:       vector.transfer_write

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
// STORE-SCATTER:        %[[COLLAPSE:.+]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1, 2]{{\]}} : memref<16x32x64xf32> into memref<32768xf32>
// STORE-SCATTER:        xegpu.store %[[VEC]], %[[COLLAPSE]][%[[IDX]]], %[[CST]] : vector<8x16x32xf32>, memref<32768xf32>, vector<8x16x32xindex>, vector<8x16x32xi1> 
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

// STORE-ND-LABEL: @no_store_masked(
// STORE-ND:       vector.transfer_write

// STORE-SCATTER-LABEL:  @no_store_masked(
// STORE-SCATTER:        vector.transfer_write
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

// STORE-ND-LABEL: @no_store_tensor(
// STORE-ND:       vector.transfer_write

// STORE-SCATTER-LABEL:  @no_store_tensor(
// STORE-SCATTER:        vector.transfer_write
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

// STORE-ND-LABEL: @no_store_non_unit_inner_stride(
// STORE-ND:       vector.transfer_write

// STORE-SCATTER-LABEL:  @no_store_non_unit_inner_stride(
// STORE-SCATTER:        vector.transfer_write
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

// STORE-ND-LABEL: @no_store_unsupported_map(
// STORE-ND:       vector.transfer_write

// STORE-SCATTER-LABEL:  @no_store_unsupported_map(
// STORE-SCATTER:        vector.transfer_write
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

// STORE-ND-LABEL: @no_store_out_of_bounds_1D_vector(
// STORE-ND:       vector.transfer_write

// STORE-SCATTER-LABEL:  @no_store_out_of_bounds_1D_vector(
// STORE-SCATTER:        vector.transfer_write
}