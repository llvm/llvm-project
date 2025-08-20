// RUN: mlir-opt %s --xevm-attach-target='module=xevm_* O=3 chip=pvc' -convert-vector-to-xegpu -split-input-file | FileCheck %s --check-prefix=LOAD-ND
// RUN: mlir-opt %s -convert-vector-to-xegpu -split-input-file | FileCheck %s --check-prefix=LOAD-GATHER

gpu.module @xevm_module {
gpu.func @load_1D_vector(%source: memref<8x16x32xf32>, %offset: index) -> vector<8xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %source[%offset, %offset, %offset], %c0
    {in_bounds = [true]} : memref<8x16x32xf32>, vector<8xf32>
  gpu.return %0 : vector<8xf32>
}

// LOAD-ND-LABEL:  @load_1D_vector(
// LOAD-ND-SAME:   %[[SRC:.+]]: memref<8x16x32xf32>,
// LOAD-ND-SAME:   %[[OFFSET:.+]]: index
// LOAD-ND:        %[[DESC:.+]] = xegpu.create_nd_tdesc
// LOAD-ND-SAME:     %[[SRC]][%[[OFFSET]], %[[OFFSET]], %[[OFFSET]]]
// LOAD-ND-SAME:     memref<8x16x32xf32> -> !xegpu.tensor_desc<8xf32,
// LOAD-ND-SAME:     boundary_check = false
// LOAD-ND:        %[[VEC:.+]] = xegpu.load_nd %[[DESC]]{{.*}}-> vector<8xf32>
// LOAD-ND:        return %[[VEC]]

// LOAD-GATHER-LABEL:  @load_1D_vector(
// LOAD-GATHER-SAME:   %[[SRC:.+]]: memref<8x16x32xf32>,
// LOAD-GATHER:        %[[CST:.+]] = arith.constant dense<true> : vector<8xi1>
// LOAD-GATHER:        %[[STEP:.+]] = vector.step : vector<8xindex>
// LOAD-GATHER-COUNT2: arith.muli {{.*}} : index
// LOAD-GATHER-COUNT2: arith.addi {{.*}} : index
// LOAD-GATHER:        %[[SPLAT:.+]] = vector.broadcast {{.*}}:  index to vector<8xindex>
// LOAD-GATHER:        %[[IDX:.+]] = arith.addi %[[SPLAT]], %[[STEP]] : vector<8xindex>
// LOAD-GATHER:        %[[COLLAPSE:.+]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1, 2]{{\]}} : memref<8x16x32xf32> into memref<4096xf32>
// LOAD-GATHER:        %[[VEC:.+]] = xegpu.load %[[COLLAPSE]]{{\[}}%[[IDX]]{{\]}}, %[[CST]] : memref<4096xf32>, vector<8xindex>, vector<8xi1> -> vector<8xf32>

}

// -----
gpu.module @xevm_module {
gpu.func @load_2D_vector(%source: memref<8x16x32xf32>,
    %offset: index) -> vector<8x16xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %source[%offset, %offset, %offset], %c0
    {in_bounds = [true, true]} : memref<8x16x32xf32>, vector<8x16xf32>
  gpu.return %0 : vector<8x16xf32>
}

// LOAD-ND-LABEL:  @load_2D_vector(
// LOAD-ND-SAME:   %[[SRC:.+]]: memref<8x16x32xf32>,
// LOAD-ND-SAME:   %[[OFFSET:.+]]: index
// LOAD-ND:        %[[DESC:.+]] = xegpu.create_nd_tdesc
// LOAD-ND-SAME:     %[[SRC]][%[[OFFSET]], %[[OFFSET]], %[[OFFSET]]]
// LOAD-ND-SAME:     memref<8x16x32xf32> -> !xegpu.tensor_desc<8x16xf32,
// LOAD-ND-SAME:     boundary_check = false
// LOAD-ND:        %[[VEC:.+]] = xegpu.load_nd %[[DESC]]{{.*}}-> vector<8x16xf32>
// LOAD-ND:        return %[[VEC]]

// LOAD-GATHER-LABEL:  @load_2D_vector(
// LOAD-GATHER-SAME:   %[[SRC:.+]]: memref<8x16x32xf32>,
// LOAD-GATHER:        %[[CST:.+]] = arith.constant dense<true> : vector<8x16xi1>
// LOAD-GATHER-COUNT2: vector.step
// LOAD-GATHER-COUNT2: vector.shape_cast
// LOAD-GATHER-COUNT2: vector.broadcast
// LOAD-GATHER-COUNT2: arith.muli {{.*}} : index
// LOAD-GATHER-COUNT2: arith.addi {{.*}} : index
// LOAD-GATHER:        %[[SPLAT:.+]] = vector.broadcast {{.*}}: index to vector<8x16xindex>
// LOAD-GATHER:        %[[IDX:.+]] = arith.addi %[[SPLAT]], {{.*}}: vector<8x16xindex>
// LOAD-GATHER:        %[[COLLAPSE:.+]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1, 2]{{\]}} : memref<8x16x32xf32> into memref<4096xf32>
// LOAD-GATHER:        %[[VEC:.+]] = xegpu.load %[[COLLAPSE]]{{\[}}%[[IDX]]{{\]}}, %[[CST]] : memref<4096xf32>, vector<8x16xindex>, vector<8x16xi1> -> vector<8x16xf32>

}


// -----
gpu.module @xevm_module {
gpu.func @load_zero_pad_out_of_bounds(%source: memref<32x64xf32>,
    %offset: index) -> vector<8x16xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %source[%offset, %offset], %c0
    {in_bounds = [false, true]} : memref<32x64xf32>, vector<8x16xf32>
  gpu.return %0 : vector<8x16xf32>
}

// LOAD-ND-LABEL:  @load_zero_pad_out_of_bounds(
// LOAD-ND-SAME:   %[[SRC:.+]]: memref<32x64xf32>,
// LOAD-ND-SAME:   %[[OFFSET:.+]]: index
// LOAD-ND:        %[[DESC:.+]] = xegpu.create_nd_tdesc %[[SRC]][%[[OFFSET]], %[[OFFSET]]]
// LOAD-ND-SAME:     memref<32x64xf32> -> !xegpu.tensor_desc<8x16xf32>
// LOAD-ND:        %[[VEC:.+]] = xegpu.load_nd %[[DESC]]{{.*}}-> vector<8x16xf32>
// LOAD-ND:        return %[[VEC]]

// LOAD-GATHER-LABEL:  @load_zero_pad_out_of_bounds(
// LOAD-GATHER:        vector.transfer_read

}


// -----
gpu.module @xevm_module {
gpu.func @load_transposed(%source: memref<32x64xf32>,
    %i: index, %j: index) -> vector<8x16xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %source[%i, %j], %c0
    {permutation_map = affine_map<(d0, d1) -> (d1, d0)>,
    in_bounds = [true, true]} : memref<32x64xf32>, vector<8x16xf32>
  gpu.return %0 : vector<8x16xf32>
}

// LOAD-ND-LABEL:  @load_transposed(
// LOAD-ND-SAME:   %[[SRC:.+]]: memref<32x64xf32>,
// LOAD-ND-SAME:   %[[OFFSET1:.+]]: index, 
// LOAD-ND-SAME:   %[[OFFSET2:.+]]: index  
// LOAD-ND:        %[[DESC:.+]] = xegpu.create_nd_tdesc %[[SRC]][%[[OFFSET1]], %[[OFFSET2]]]
// LOAD-ND-SAME:     memref<32x64xf32> -> !xegpu.tensor_desc<16x8xf32
// LOAD-ND:        %[[VEC:.+]] = xegpu.load_nd %[[DESC]] <{transpose = array<i64: 1, 0>}>
// LOAD-ND-SAME:     -> vector<8x16xf32>
// LOAD-ND:        return %[[VEC]]


// LOAD-GATHER-LABEL:  @load_transposed(
// LOAD-GATHER-SAME:    %[[SRC:.+]]: memref<32x64xf32>,
// LOAD-GATHER:         %[[CST:.+]] = arith.constant dense<true> : vector<8x16xi1>
// LOAD-GATHER-COUNT2:  vector.step
// LOAD-GATHER-COUNT2:  vector.shape_cast
// LOAD-GATHER-COUNT2: vector.broadcast
// LOAD-GATHER-COUNT2: arith.muli {{.*}} : index
// LOAD-GATHER-COUNT2: arith.addi {{.*}} : index
// LOAD-GATHER:        %[[BCAST2:.+]] = vector.broadcast {{.*}} : index to vector<8x16xindex>
// LOAD-GATHER:        %[[IDX:.+]] = arith.addi %[[BCAST2]], {{.*}}: vector<8x16xindex>
// LOAD-GATHER:        %[[COLLAPSE:.*]] = memref.collapse_shape %arg0 {{\[\[}}0, 1{{\]\]}} : memref<32x64xf32> into memref<2048xf32>
// LOAD-GATHER:        %[[LOAD:.*]] = xegpu.load %[[COLLAPSE]][%[[IDX]]], %[[CST]] : memref<2048xf32>, vector<8x16xindex>, vector<8x16xi1> -> vector<8x16xf32>

}

// -----
gpu.module @xevm_module {
gpu.func @load_dynamic_source(%source: memref<?x?x?xf32>,
    %i: index, %j: index, %k: index) -> vector<8x16xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %source[%i, %j, %k], %c0
    {in_bounds = [true, true]} : memref<?x?x?xf32>, vector<8x16xf32>
  gpu.return %0 : vector<8x16xf32>
}
// LOAD-ND-LABEL:  @load_dynamic_source(
// LOAD-ND-SAME:   %[[SRC:.+]]: memref<?x?x?xf32>,
// LOAD-ND-SAME:   %[[OFFSET:.+]]: index
// LOAD-ND:        %[[C2:.+]] = arith.constant 2 : index
// LOAD-ND:        %[[C1:.+]] = arith.constant 1 : index
// LOAD-ND:        %[[C0:.+]] = arith.constant 0 : index
// LOAD-ND-DAG:    %[[DIM_0:.+]] = memref.dim %[[SRC]], %[[C0]]
// LOAD-ND-DAG:    %[[DIM_1:.+]] = memref.dim %[[SRC]], %[[C1]]
// LOAD-ND-DAG:    %[[DIM_2:.+]] = memref.dim %[[SRC]], %[[C2]]
// LOAD-ND:        %[[DIM_0_STRIDE:.+]] = arith.muli %[[DIM_2]], %[[DIM_1]]
// LOAD-ND:        %[[DESC:.+]] = xegpu.create_nd_tdesc %[[SRC]][%[[OFFSET:.+]], %[[OFFSET:.+]], %[[OFFSET:.+]]]
// LOAD-ND:        %[[VEC:.+]] = xegpu.load_nd %[[DESC]]{{.*}}-> vector<8x16xf32>
// LOAD-ND:        return %[[VEC]]


// LOAD-GATHER-LABEL:  @load_dynamic_source(
// LOAD-GATHER-SAME:   %[[ARG0:.+]]: memref<?x?x?xf32>,
// LOAD-GATHER:        %[[CST:.+]] = arith.constant dense<true> : vector<8x16xi1>
// LOAD-GATHER:        memref.extract_strided_metadata %[[ARG0]]
// LOAD-GATHER-COUNT2: vector.step
// LOAD-GATHER-COUNT2: vector.shape_cast
// LOAD-GATHER-COUNT2: vector.broadcast
// LOAD-GATHER-COUNT2: arith.muli {{.*}} : index
// LOAD-GATHER-COUNT2: arith.addi {{.*}} : index
// LOAD-GATHER:        %[[BROADIDX:.+]] = vector.broadcast {{.*}} : index to vector<8x16xindex>
// LOAD-GATHER:        %[[FINALIDX:.+]] = arith.addi %[[BROADIDX]], {{.*}} : vector<8x16xindex>
// LOAD-GATHER:        %[[COLLAPSE:.+]] = memref.collapse_shape %[[ARG0]] {{\[}}[0, 1, 2]{{\]}} : memref<?x?x?xf32> into memref<?xf32>
// LOAD-GATHER:        %[[RES:.+]] = xegpu.load %[[COLLAPSE]][%[[FINALIDX]]], %[[CST]] : memref<?xf32>, vector<8x16xindex>, vector<8x16xi1> -> vector<8x16xf32>
// LOAD-GATHER:        gpu.return %[[RES]] : vector<8x16xf32>
}

// -----
gpu.module @xevm_module {
gpu.func @load_dynamic_source2(%source: memref<?x8x16xf32>,
    %i: index, %j: index, %k: index) -> vector<8x16xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %source[%i, %j, %k], %c0
    {in_bounds = [true, true]} : memref<?x8x16xf32>, vector<8x16xf32>
  gpu.return %0 : vector<8x16xf32>
}

// LOAD-ND-LABEL:  @load_dynamic_source2(
// LOAD-ND-DAG:    %[[C0:.+]] = arith.constant 0 : index
// LOAD-ND-DAG:    %[[DIM:.+]] = memref.dim %{{.*}}, %[[C0]] : memref<?x8x16xf32>
// LOAD-ND:        %[[DESC:.+]] = xegpu.create_nd_tdesc %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}], shape : [%[[DIM]], 8, 16], strides : [128, 16, 1] : memref<?x8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
// LOAD-ND:        %[[VEC:.+]] = xegpu.load_nd %[[DESC]] : !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<8x16xf32>
// LOAD-ND:        return %[[VEC]] : vector<8x16xf32>

// LOAD-GATHER-LABEL:  @load_dynamic_source2(
// LOAD-GATHER-DAG:    %[[CST_0:.+]] = arith.constant dense<true> : vector<8x16xi1>
// LOAD-GATHER-COUNT2: vector.step
// LOAD-GATHER-COUNT2: vector.shape_cast
// LOAD-GATHER-COUNT2: vector.broadcast
// LOAD-GATHER-COUNT2: arith.muli {{.*}} : index
// LOAD-GATHER-COUNT2: arith.addi {{.*}} : index
// LOAD-GATHER-DAG:    %[[BCASTIDX:.+]] = vector.broadcast {{.*}} : index to vector<8x16xindex>
// LOAD-GATHER-DAG:    %[[OFFSETS:.+]] = arith.addi %[[BCASTIDX]], {{.*}} : vector<8x16xindex>
// LOAD-GATHER-DAG:    %[[COLLAPSE:.+]] = memref.collapse_shape %arg0 {{\[}}[0, 1, 2]{{\]}} : memref<?x8x16xf32> into memref<?xf32>
// LOAD-GATHER:        %[[VEC:.+]] = xegpu.load %[[COLLAPSE]]{{\[}}%[[OFFSETS]]{{\]}}, %[[CST_0]] : memref<?xf32>, vector<8x16xindex>, vector<8x16xi1> -> vector<8x16xf32> 

}

// -----
gpu.module @xevm_module {
gpu.func @load_dynamic_source3(%source: memref<?x?x?x?x?xf32>,
    %i: index, %j: index, %k: index, %l: index, %m: index) -> vector<2x4x8x16xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %source[%i, %j, %k, %l, %m], %c0
    {in_bounds = [true, true, true, true]} : memref<?x?x?x?x?xf32>, vector<2x4x8x16xf32>
  gpu.return %0 : vector<2x4x8x16xf32>
}

// LOAD-ND-LABEL:  @load_dynamic_source3(
// LOAD-ND:        vector.transfer_read

// LOAD-GATHER-LABEL:  @load_dynamic_source3(
// LOAD-GATHER-SAME:   %[[SRC:.+]]: memref<?x?x?x?x?xf32>
// LOAD-GATHER:        %[[CST:.+]] = arith.constant dense<true> : vector<2x4x8x16xi1>
// LOAD-GATHER:        memref.extract_strided_metadata %[[SRC]] : memref<?x?x?x?x?xf32> -> memref<f32>, index, index, index, index, index, index, index, index, index, index, index
// LOAD-GATHER-COUNT4: vector.step
// LOAD-GATHER-COUNT3: vector.broadcast
// LOAD-GATHER-COUNT4: vector.shape_cast
// LOAD-GATHER-COUNT4: vector.broadcast {{.*}} : vector<2x4x8x16xindex>
// LOAD-GATHER-COUNT3: arith.addi {{.*}} : vector<2x4x8x16xindex>
// LOAD-GATHER:        %[[SPLAT:.+]] = vector.broadcast {{.*}} : index to vector<2x4x8x16xindex>
// LOAD-GATHER:        %[[IDX:.+]] = arith.addi %[[SPLAT]], {{.*}} : vector<2x4x8x16xindex>
// LOAD-GATHER:        %[[COLLAPSE:.+]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1, 2, 3, 4]{{\]}} : memref<?x?x?x?x?xf32> into memref<?xf32>
// LOAD-GATHER:        %[[VEC:.+]] = xegpu.load %[[COLLAPSE]]{{\[}}%[[IDX]]{{\]}}, %[[CST]] : memref<?xf32>, vector<2x4x8x16xindex>, vector<2x4x8x16xi1> -> vector<2x4x8x16xf32>
// LOAD-GATHER:        return %[[VEC]]
}

// -----
gpu.module @xevm_module {
gpu.func @load_high_dim_vector(%source: memref<16x32x64xf32>,
    %offset: index, %arg2: index) -> vector<8x16x32xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %source[%offset, %arg2, %offset], %c0
    {in_bounds = [true, true, true]} : memref<16x32x64xf32>, vector<8x16x32xf32>
  gpu.return %0 : vector<8x16x32xf32>
}

// LOAD-ND-LABEL:  @load_high_dim_vector(
// LOAD-ND:        vector.transfer_read

// LOAD-GATHER-LABEL:  @load_high_dim_vector(
// LOAD-GATHER:        %[[CST:.+]] = arith.constant dense<true> : vector<8x16x32xi1>
// LOAD-GATHER:        %[[CST_0:.+]] = arith.constant dense<64> : vector<16xindex>
// LOAD-GATHER:        %[[CST_1:.+]] = arith.constant dense<2048> : vector<8xindex>
// LOAD-GATHER:        %[[C2048:.+]] = arith.constant 2048 : index
// LOAD-GATHER:        %[[C64:.+]] = arith.constant 64 : index
// LOAD-GATHER-COUNT3: vector.step
// LOAD-GATHER-COUNT3: vector.shape_cast
// LOAD-GATHER-COUNT3: vector.broadcast {{.*}} : vector<8x16x32xindex>
// LOAD-GATHER-COUNT2: arith.addi {{.*}} : vector<8x16x32xindex>
// LOAD-GATHER:        %[[BCASTOFF:.+]] = vector.broadcast {{.*}} : index to vector<8x16x32xindex>
// LOAD-GATHER:        %[[IDX:.+]] = arith.addi %[[BCASTOFF]], {{.*}} : vector<8x16x32xindex>
// LOAD-GATHER:        %[[COLLAPSE:.+]] = memref.collapse_shape %arg0 {{\[}}[0, 1, 2]{{\]}} : memref<16x32x64xf32> into memref<32768xf32>
// LOAD-GATHER:        %[[VEC:.+]] = xegpu.load %[[COLLAPSE]][%[[IDX]]], %[[CST]] : memref<32768xf32>, vector<8x16x32xindex>, vector<8x16x32xi1> -> vector<8x16x32xf32>

}

// -----
gpu.module @xevm_module {
gpu.func @load_transpose_f16(%source: memref<32x64xf16>,
    %offset: index) -> vector<8x16xf16> {
  %c0 = arith.constant 0.0 : f16
  %0 = vector.transfer_read %source[%offset, %offset], %c0
    {permutation_map = affine_map<(d0, d1) -> (d1, d0)>,
    in_bounds = [true, true]} : memref<32x64xf16>, vector<8x16xf16>
  gpu.return %0 : vector<8x16xf16>
}

// LOAD-ND-LABEL:  @load_transpose_f16(
// LOAD-ND:        vector.transfer_read

// LOAD-GATHER-LABEL:  @load_transpose_f16(
// LOAD-GATHER-SAME:    %[[SRC:.+]]: memref<32x64xf16>,
// LOAD-GATHER:         %[[CST:.+]] = arith.constant dense<true> : vector<8x16xi1>
// LOAD-GATHER-COUNT2:  vector.step
// LOAD-GATHER-COUNT2:  vector.shape_cast
// LOAD-GATHER-COUNT2: vector.broadcast
// LOAD-GATHER-COUNT2: arith.muli {{.*}} : index
// LOAD-GATHER-COUNT2: arith.addi {{.*}} : index
// LOAD-GATHER:        %[[BCAST2:.+]] = vector.broadcast {{.*}} : index to vector<8x16xindex>
// LOAD-GATHER:        %[[IDX:.+]] = arith.addi %[[BCAST2]], {{.*}}: vector<8x16xindex>
// LOAD-GATHER:        %[[COLLAPSE:.*]] = memref.collapse_shape %arg0 {{\[\[}}0, 1{{\]\]}} : memref<32x64xf16> into memref<2048xf16>
// LOAD-GATHER:        %[[LOAD:.*]] = xegpu.load %[[COLLAPSE]][%[[IDX]]], %[[CST]] : memref<2048xf16>, vector<8x16xindex>, vector<8x16xi1> -> vector<8x16xf16>
}

// -----
gpu.module @xevm_module {
gpu.func @no_load_out_of_bounds_non_zero_pad(%source: memref<32x64xf32>,
    %offset: index, %arg2: index, %pad: f32) -> (vector<8x16xf32>, vector<8x16xf32>) {
  %c1 = arith.constant 1.0 : f32
  %0 = vector.transfer_read %source[%offset, %arg2], %c1
    {in_bounds = [true, false]} : memref<32x64xf32>, vector<8x16xf32>
  %1 = vector.transfer_read %source[%arg2, %offset], %pad
    {in_bounds = [false, true]} : memref<32x64xf32>, vector<8x16xf32>
  gpu.return %0, %1 : vector<8x16xf32>, vector<8x16xf32>
}

// LOAD-ND-LABEL:    @no_load_out_of_bounds_non_zero_pad(
// LOAD-ND-COUNT-2: vector.transfer_read

// LOAD-GATHER-LABEL: @no_load_out_of_bounds_non_zero_pad(
// LOAD-GATHER-COUNT-2: vector.transfer_read
}

// -----
gpu.module @xevm_module {
gpu.func @no_load_out_of_bounds_1D_vector(%source: memref<8x16x32xf32>,
    %offset: index) -> vector<8xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %source[%offset, %offset, %offset], %c0
    {in_bounds = [false]} : memref<8x16x32xf32>, vector<8xf32>
  gpu.return %0 : vector<8xf32>
}

// LOAD-ND-LABEL:  @no_load_out_of_bounds_1D_vector(
// LOAD-ND:        vector.transfer_read

// LOAD-GATHER-LABEL:  @no_load_out_of_bounds_1D_vector(
// LOAD-GATHER:        vector.transfer_read
}

// -----
gpu.module @xevm_module {
gpu.func @no_load_masked(%source : memref<4xf32>,
    %offset : index) -> vector<4xf32> {
  %c0 = arith.constant 0.0 : f32
  %mask = arith.constant dense<[0, 1, 0, 1]> : vector<4xi1>
  %0 = vector.transfer_read %source[%offset], %c0, %mask
    {in_bounds = [true]} : memref<4xf32>, vector<4xf32>
  gpu.return %0 : vector<4xf32>
}

// LOAD-ND-LABEL:  @no_load_masked(
// LOAD-ND:        vector.transfer_read

// LOAD-GATHER-LABEL:  @no_load_masked(
// LOAD-GATHER:        vector.transfer_read
}

// -----
gpu.module @xevm_module {
gpu.func @no_load_tensor(%source: tensor<32x64xf32>,
    %offset: index, %arg2: index) -> vector<8x16xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %source[%offset, %arg2], %c0
    {in_bounds = [true, true]} : tensor<32x64xf32>, vector<8x16xf32>
  gpu.return %0 : vector<8x16xf32>
}

// LOAD-ND-LABEL:  @no_load_tensor(
// LOAD-ND:        vector.transfer_read

// LOAD-GATHER-LABEL:  @no_load_tensor(
// LOAD-GATHER:        vector.transfer_read
}


// -----
gpu.module @xevm_module {
gpu.func @no_load_non_unit_inner_stride(
    %source: memref<32xf32, strided<[?], offset: ?>>,
    %offset: index) -> vector<8xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %source[%offset], %c0 {in_bounds = [true]}
    : memref<32xf32, strided<[?], offset: ?>>, vector<8xf32>
  gpu.return %0 : vector<8xf32>
}

// LOAD-ND-LABEL:  @no_load_non_unit_inner_stride(
// LOAD-ND:        vector.transfer_read

// LOAD-GATHER-LABEL:  @no_load_non_unit_inner_stride(
// LOAD-GATHER:        vector.transfer_read
}


// -----
gpu.module @xevm_module {
gpu.func @no_load_unsupported_map(%source: memref<16x32x64xf32>,
    %offset: index) -> vector<8x16xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %source[%offset, %offset, %offset], %c0
    {permutation_map = affine_map<(d0, d1, d2) -> (d0, d2)>,
    in_bounds = [true, true]} : memref<16x32x64xf32>, vector<8x16xf32>
  gpu.return %0 : vector<8x16xf32>
}

// LOAD-ND-LABEL:  @no_load_unsupported_map(
// LOAD-ND:        vector.transfer_read

// LOAD-GATHER-LABEL:  @no_load_unsupported_map(
// LOAD-GATHER:        vector.transfer_read
}

