// RUN: mlir-opt %s --xevm-attach-target='module=xevm_* O=3 chip=pvc' -convert-vector-to-xegpu -split-input-file | FileCheck %s --check-prefixes=LOAD-ND,CHECK
// RUN: mlir-opt %s -convert-vector-to-xegpu -split-input-file | FileCheck %s --check-prefixes=LOAD-GATHER,CHECK

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
// LOAD-ND:        %[[ELEM_BYTES:.+]] = arith.constant 4 : index
// LOAD-ND:        %[[COLLAPSED:.+]] = memref.subview %[[SRC]][%[[OFFSET]], %[[OFFSET]], 0]
// LOAD-ND:        %[[BASE_BUFFER:.*]], %[[OFF1:.*]], %[[SIZES:.*]], %[[STRIDES:.*]] = memref.extract_strided_metadata %[[COLLAPSED]]
// LOAD-ND-SAME:     : memref<32xf32, strided<[1], offset: ?>> -> memref<f32>, index, index, index
// LOAD-ND:        %[[INTPTR:.*]] = memref.extract_aligned_pointer_as_index %[[BASE_BUFFER]]
// LOAD-ND-SAME:     : memref<f32> -> index
// LOAD-ND:        %[[MUL:.*]] = arith.muli %[[OFF1]], %[[ELEM_BYTES]] : index
// LOAD-ND:        %[[ADD:.*]] = arith.addi %[[INTPTR]], %[[MUL]] : index
// LOAD-ND:        %[[I64PTR:.*]] = arith.index_cast %[[ADD]] : index to i64
// LOAD-ND:        %[[DESC:.+]] = xegpu.create_nd_tdesc %[[I64PTR]], shape : [32],
// LOAD-ND-SAME:                   strides : [1] : i64 -> !xegpu.tensor_desc<8xf32,
// LOAD-ND-SAME:     #xegpu.block_tdesc_attr<boundary_check = false>>
// LOAD-ND:        %[[VEC:.+]] = xegpu.load_nd %[[DESC]][%[[OFFSET]]]
// LOAD-ND-SAME:     : !xegpu.tensor_desc<8xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<8xf32>

// LOAD-GATHER-LABEL:  @load_1D_vector(
// LOAD-GATHER-SAME:   %[[SRC:.+]]: memref<8x16x32xf32>,
// LOAD-GATHER:        %[[CST:.+]] = arith.constant dense<true> : vector<8xi1>
// LOAD-GATHER:        %[[STEP:.+]] = vector.step : vector<8xindex>
// LOAD-GATHER-COUNT2: arith.muli {{.*}} : index
// LOAD-GATHER-COUNT2: arith.addi {{.*}} : index
// LOAD-GATHER:        %[[SPLAT:.+]] = vector.broadcast {{.*}}:  index to vector<8xindex>
// LOAD-GATHER:        %[[IDX:.+]] = arith.addi %[[SPLAT]], %[[STEP]] : vector<8xindex>
// LOAD-GATHER:        %[[COLLAPSE:.+]] = memref.extract_aligned_pointer_as_index %[[SRC]] : memref<8x16x32xf32> -> index
// LOAD-GATHER:        %[[COLLAPSE_I:.+]] = arith.index_cast %[[COLLAPSE]] : index to i64
// LOAD-GATHER:        %[[VEC:.+]] = xegpu.load %[[COLLAPSE_I]]{{\[}}%[[IDX]]{{\]}}, %[[CST]] : i64, vector<8xindex>, vector<8xi1> -> vector<8xf32>

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
// LOAD-ND:        %[[ELEM_BYTES:.+]] = arith.constant 4 : index
// LOAD-ND:        %[[COLLAPSED:.+]] = memref.subview %[[SRC]][%[[OFFSET]], 0, 0]
// LOAD-ND:        %[[BASE_BUFFER:.*]], %[[OFF1:.*]], %[[SIZES:.*]]:2, %[[STRIDES:.*]]:2 = memref.extract_strided_metadata %[[COLLAPSED]]
// LOAD-ND:        %[[INTPTR:.*]] = memref.extract_aligned_pointer_as_index %[[BASE_BUFFER]]
// LOAD-ND-SAME:     : memref<f32> -> index
// LOAD-ND:        %[[MUL:.*]] = arith.muli %[[OFF1]], %[[ELEM_BYTES]] : index
// LOAD-ND:        %[[ADD:.*]] = arith.addi %[[INTPTR]], %[[MUL]] : index
// LOAD-ND:        %[[I64PTR:.*]] = arith.index_cast %[[ADD]] : index to i64
// LOAD-ND:        %[[DESC:.+]] = xegpu.create_nd_tdesc %[[I64PTR]], shape : [16, 32],
// LOAD-ND-SAME:                   strides : [32, 1] : i64 -> !xegpu.tensor_desc<8x16xf32,
// LOAD-ND-SAME:     boundary_check = false
// LOAD-ND:        %[[VEC:.+]] = xegpu.load_nd %[[DESC]][%[[OFFSET]], %[[OFFSET]]]{{.*}}-> vector<8x16xf32>
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
// LOAD-GATHER:        %[[COLLAPSE:.+]] = memref.extract_aligned_pointer_as_index %[[SRC]] : memref<8x16x32xf32> -> index
// LOAD-GATHER:        %[[COLLAPSE_I:.+]] = arith.index_cast %[[COLLAPSE]] : index to i64
// LOAD-GATHER:        %[[VEC:.+]] = xegpu.load %[[COLLAPSE_I]]{{\[}}%[[IDX]]{{\]}}, %[[CST]] : i64, vector<8x16xindex>, vector<8x16xi1> -> vector<8x16xf32>

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
// LOAD-ND:        %[[DESC:.+]] = xegpu.create_nd_tdesc %[[SRC]]
// LOAD-ND-SAME:     memref<32x64xf32> -> !xegpu.tensor_desc<8x16xf32>
// LOAD-ND:        %[[VEC:.+]] = xegpu.load_nd %[[DESC]][%[[OFFSET]], %[[OFFSET]]]{{.*}}-> vector<8x16xf32>
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
// LOAD-ND:        %[[DESC:.+]] = xegpu.create_nd_tdesc %[[SRC]]
// LOAD-ND-SAME:     memref<32x64xf32> -> !xegpu.tensor_desc<16x8xf32
// LOAD-ND:        %[[VEC:.+]] = xegpu.load_nd %[[DESC]][%[[OFFSET1]], %[[OFFSET2]]]
// LOAD-ND-SAME:     -> vector<16x8xf32>
// LOAD-ND:        %[[VEC_TRANSPOSED:.+]] = vector.transpose %[[VEC]], [1, 0] : vector<16x8xf32> to vector<8x16xf32>
// LOAD-ND:        return %[[VEC_TRANSPOSED]]


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
// LOAD-GATHER:        %[[COLLAPSE:.*]] = memref.extract_aligned_pointer_as_index %arg0 : memref<32x64xf32> -> index
// LOAD-GATHER:        %[[COLLAPSE_I:.+]] = arith.index_cast %[[COLLAPSE]] : index to i64
// LOAD-GATHER:        %[[LOAD:.*]] = xegpu.load %[[COLLAPSE_I]][%[[IDX]]], %[[CST]] : i64, vector<8x16xindex>, vector<8x16xi1> -> vector<8x16xf32>

}

// -----
gpu.module @xevm_module {
gpu.func @load_transpose_3d_memref(%source: memref<32x64x128xf32>,
    %i: index, %j: index, %k: index) -> vector<8x16xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %source[%i, %j, %k], %c0
    {permutation_map = affine_map<(d0, d1, d2) -> (d2, d1)>,
    in_bounds = [true, true]} : memref<32x64x128xf32>, vector<8x16xf32>
  gpu.return %0 : vector<8x16xf32>
}

// LOAD-ND-LABEL:  @load_transpose_3d_memref(
// LOAD-ND-SAME:   %[[SRC:.+]]: memref<32x64x128xf32>,
// LOAD-ND-SAME:   %[[OFF0:.+]]: index, %[[OFF1:.+]]: index, %[[OFF2:.+]]: index) -> vector<8x16xf32> {
// LOAD-ND:        %[[ELEM_BYTES:.+]] = arith.constant 4 : index
// LOAD-ND:        %[[COLLAPSED:.+]] = memref.subview %[[SRC]][%[[OFF0]], 0, 0]
// LOAD-ND:        %[[BASE_BUFFER:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:2, %[[STRIDES:.*]]:2 = memref.extract_strided_metadata %[[COLLAPSED]]
// LOAD-ND:        %[[INTPTR:.*]] = memref.extract_aligned_pointer_as_index %[[BASE_BUFFER]]
// LOAD-ND-SAME:     : memref<f32> -> index
// LOAD-ND:        %[[MUL:.*]] = arith.muli %[[OFFSET]], %[[ELEM_BYTES]] : index
// LOAD-ND:        %[[ADD:.*]] = arith.addi %[[INTPTR]], %[[MUL]] : index
// LOAD-ND:        %[[I64PTR:.*]] = arith.index_cast %[[ADD]] : index to i64
// LOAD-ND:        %[[DESC:.+]] = xegpu.create_nd_tdesc %[[I64PTR]], shape : [64, 128],
// LOAD-ND-SAME:                   strides : [128, 1] : i64 -> !xegpu.tensor_desc<16x8xf32,
// LOAD-ND-SAME:     #xegpu.block_tdesc_attr<boundary_check = false>>
// LOAD-ND:        %[[VEC:.+]] = xegpu.load_nd %[[DESC]][%[[OFF1]], %[[OFF2]]]
// LOAD-ND-SAME:     : !xegpu.tensor_desc<16x8xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<16x8xf32>
// LOAD-ND:        %[[VEC_TRANSPOSED:.+]] = vector.transpose %[[VEC]], [1, 0] : vector<16x8xf32> to vector<8x16xf32>

// LOAD-GATHER-LABEL:  @load_transpose_3d_memref(
// LOAD-GATHER-SAME:    %[[SRC:.+]]: memref<32x64x128xf32>,
// LOAD-GATHER-SAME:    %[[OFF1:.+]]: index, %[[OFF2:.+]]: index, %[[OFF3:.+]]: index
// LOAD-GATHER:         %[[BCAST3:.+]] = vector.broadcast %{{.*}} : index to vector<8x16xindex>
// LOAD-GATHER:         %[[IDX:.+]] = arith.addi %[[BCAST3]], %{{.*}} : vector<8x16xindex>
// LOAD-GATHER:         %[[INTPTR:.*]] = memref.extract_aligned_pointer_as_index %[[SRC]] : memref<32x64x128xf32> -> index
// LOAD-GATHER-NEXT:    %[[I64PTR:.+]] = arith.index_cast %[[INTPTR]] : index to i64
// LOAD-GATHER-NEXT:    %[[LOAD:.*]] = xegpu.load %[[I64PTR]][%[[IDX]]], %{{.*}} : i64, vector<8x16xindex>, vector<8x16xi1> -> vector<8x16xf32>

}

// -----
// A high-dim load whose innermost two dims are transposed lowers to an
// (untransposed) nd block load followed by a vector.transpose of the last two
// dims.
gpu.module @xevm_module {
gpu.func @load_high_dim_transposed(%source: memref<2x2x64x128xf16>,
    %offset: index) -> vector<1x1x64x128xf16> {
  %c0 = arith.constant 0.0 : f16
  %0 = vector.transfer_read %source[%offset, %offset, %offset, %offset], %c0
    {permutation_map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>,
    in_bounds = [true, true, true, true]}
    : memref<2x2x64x128xf16>, vector<1x1x64x128xf16>
  gpu.return %0 : vector<1x1x64x128xf16>
}

// LOAD-ND-LABEL:  @load_high_dim_transposed(
// LOAD-ND-SAME:   %[[SRC:.+]]: memref<2x2x64x128xf16>,
// LOAD-ND:        %[[DESC:.+]] = xegpu.create_nd_tdesc %[[SRC]] : memref<2x2x64x128xf16>
// LOAD-ND-SAME:     -> !xegpu.tensor_desc<1x1x128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
// LOAD-ND:        %[[VEC:.+]] = xegpu.load_nd %[[DESC]]
// LOAD-ND-SAME:     -> vector<1x1x128x64xf16>
// LOAD-ND:        vector.transpose %[[VEC]], [0, 1, 3, 2] : vector<1x1x128x64xf16> to vector<1x1x64x128xf16>

// LOAD-GATHER-LABEL:  @load_high_dim_transposed(
// LOAD-GATHER:        %[[VEC:.+]] = xegpu.load {{.*}} : i64, vector<1x1x64x128xindex>, vector<1x1x64x128xi1> -> vector<1x1x64x128xf16>
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
// LOAD-ND-SAME:   %[[OFF0:.+]]: index, %[[OFF1:.+]]: index, %[[OFF2:.+]]: index
// LOAD-ND:        %[[ELEM_BYTES:.+]] = arith.constant 4 : index
// LOAD-ND:        %[[COLLAPSED:.+]] = memref.subview %[[SRC]][%[[OFF0]], 0, 0]
// LOAD-ND:        %[[BASE_BUFFER:.*]], %[[OFFSET:.*]], %[[SIZES:.+]]:2, %[[STRIDES:.+]]:2 = memref.extract_strided_metadata %[[COLLAPSED]]
// LOAD-ND:        %[[INTPTR:.*]] = memref.extract_aligned_pointer_as_index %[[BASE_BUFFER]] : memref<f32> -> index
// LOAD-ND:        %[[MUL:.*]] = arith.muli %[[OFFSET]], %[[ELEM_BYTES]] : index
// LOAD-ND:        %[[ADD:.*]] = arith.addi %[[INTPTR]], %[[MUL]] : index
// LOAD-ND:        %[[I64PTR:.*]] = arith.index_cast %[[ADD]] : index to i64
// LOAD-ND:        %[[DESC:.+]] = xegpu.create_nd_tdesc %[[I64PTR]], shape : [%[[SIZES]]#0, %[[SIZES]]#1],
// LOAD-ND-SAME:                    strides : [%[[STRIDES]]#0, 1] : i64 -> !xegpu.tensor_desc<8x16xf32,
// LOAD-ND-SAME:                      #xegpu.block_tdesc_attr<boundary_check = false>>
// LOAD-ND:        %[[VEC:.+]] = xegpu.load_nd %[[DESC]][%[[OFF1]], %[[OFF2]]]{{.*}}-> vector<8x16xf32>
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
// LOAD-GATHER:        %[[COLLAPSE:.+]] = memref.extract_aligned_pointer_as_index %[[ARG0]] : memref<?x?x?xf32> -> index
// LOAD-GATHER:        %[[COLLAPSE_I:.+]] = arith.index_cast %[[COLLAPSE]] : index to i64
// LOAD-GATHER:        %[[RES:.+]] = xegpu.load %[[COLLAPSE_I]]{{\[}}%[[FINALIDX]]{{\]}}, %[[CST]] : i64, vector<8x16xindex>, vector<8x16xi1> -> vector<8x16xf32>
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
// LOAD-ND-SAME:   %[[SRC:.+]]: memref<?x8x16xf32>,
// LOAD-ND-SAME:   %[[OFF0:.+]]: index, %[[OFF1:.+]]: index, %[[OFF2:.+]]: index
// LOAD-ND:        %[[ELEM_BYTES:.+]] = arith.constant 4 : index
// LOAD-ND:        %[[COLLAPSED:.+]] = memref.subview %[[SRC]][%[[OFF0]], 0, 0]
// LOAD-ND:        %[[BASE_BUFFER:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:2, %[[STRIDES:.*]]:2 = memref.extract_strided_metadata %[[COLLAPSED]]
// LOAD-ND:        %[[INTPTR:.*]] = memref.extract_aligned_pointer_as_index %[[BASE_BUFFER]]
// LOAD-ND:        %[[MUL:.*]] = arith.muli %[[OFFSET]], %[[ELEM_BYTES]] : index
// LOAD-ND:        %[[ADD:.*]] = arith.addi %[[INTPTR]], %[[MUL]] : index
// LOAD-ND:        %[[I64PTR:.*]] = arith.index_cast %[[ADD]] : index to i64
// LOAD-ND:        %[[DESC:.*]] = xegpu.create_nd_tdesc %[[I64PTR]], shape : [8, 16], strides : [16, 1] :
// LOAD-ND-SAME:                    i64 -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
// LOAD-ND:        %[[VEC:.+]] = xegpu.load_nd %[[DESC]][%{{.*}}, %{{.*}}] : !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<8x16xf32>
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
// LOAD-GATHER-DAG:    %[[COLLAPSE:.+]] = memref.extract_aligned_pointer_as_index %arg0 : memref<?x8x16xf32> -> index
// LOAD-GATHER-DAG:    %[[COLLAPSE_I:.+]] = arith.index_cast %[[COLLAPSE]] : index to i64
// LOAD-GATHER:        %[[VEC:.+]] = xegpu.load %[[COLLAPSE_I]]{{\[}}%[[OFFSETS]]{{\]}}, %[[CST_0]] : i64, vector<8x16xindex>, vector<8x16xi1> -> vector<8x16xf32>

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
// LOAD-ND-SAME:   %[[SRC:.+]]: memref<?x?x?x?x?xf32>
// LOAD-ND:        %[[SUBVIEW:.+]] = memref.subview %[[SRC]]
// LOAD-ND:        %[[BASE_BUFFER:.*]], %[[OFF1:.*]], %[[SIZES:.*]]:4, %[[STRIDES:.*]]:4 = memref.extract_strided_metadata %[[SUBVIEW]]
// LOAD-ND:        %[[DESC:.+]] = xegpu.create_nd_tdesc
// LOAD-ND-SAME:     -> !xegpu.tensor_desc<2x4x8x16xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
// LOAD-ND:        %[[VEC:.+]] = xegpu.load_nd %[[DESC]]
// LOAD-ND-SAME:     -> vector<2x4x8x16xf32>
// LOAD-ND:        return %[[VEC]]

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
// LOAD-GATHER:        %[[COLLAPSE:.+]] = memref.extract_aligned_pointer_as_index %[[SRC]] : memref<?x?x?x?x?xf32> -> index
// LOAD-GATHER:        %[[COLLAPSE_I:.+]] = arith.index_cast %[[COLLAPSE]] : index to i64
// LOAD-GATHER:        %[[VEC:.+]] = xegpu.load %[[COLLAPSE_I]]{{\[}}%[[IDX]]{{\]}}, %[[CST]] : i64, vector<2x4x8x16xindex>, vector<2x4x8x16xi1> -> vector<2x4x8x16xf32>
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
// LOAD-ND:        %[[DESC:.+]] = xegpu.create_nd_tdesc %arg0 : memref<16x32x64xf32>
// LOAD-ND-SAME:     -> !xegpu.tensor_desc<8x16x32xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
// LOAD-ND:        %[[VEC:.+]] = xegpu.load_nd %[[DESC]]
// LOAD-ND-SAME:     -> vector<8x16x32xf32>

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
// LOAD-GATHER:        %[[COLLAPSE:.+]] = memref.extract_aligned_pointer_as_index %arg0 : memref<16x32x64xf32> -> index
// LOAD-GATHER:        %[[COLLAPSE_I:.+]] = arith.index_cast %[[COLLAPSE]] : index to i64
// LOAD-GATHER:        %[[VEC:.+]] = xegpu.load %[[COLLAPSE_I]][%[[IDX]]], %[[CST]] : i64, vector<8x16x32xindex>, vector<8x16x32xi1> -> vector<8x16x32xf32>

}

// -----
gpu.module @xevm_module {
gpu.func @load_8D_vector(%source: memref<2x2x2x2x2x2x2x2xf32>,
    %offset: index) -> vector<2x2x2x2x2x2x2x2xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %source[%offset, %offset, %offset, %offset, %offset, %offset, %offset, %offset], %c0
    {in_bounds = [true, true, true, true, true, true, true, true]} : memref<2x2x2x2x2x2x2x2xf32>, vector<2x2x2x2x2x2x2x2xf32>
  gpu.return %0 : vector<2x2x2x2x2x2x2x2xf32>
}

// LOAD-ND-LABEL:  @load_8D_vector(
// LOAD-ND-SAME:   %[[SRC:.+]]: memref<2x2x2x2x2x2x2x2xf32>,
// LOAD-ND:        %[[DESC:.+]] = xegpu.create_nd_tdesc %[[SRC]] : memref<2x2x2x2x2x2x2x2xf32>
// LOAD-ND-SAME:     -> !xegpu.tensor_desc<2x2x2x2x2x2x2x2xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
// LOAD-ND:        %[[VEC:.+]] = xegpu.load_nd %[[DESC]]
// LOAD-ND-SAME:     -> vector<2x2x2x2x2x2x2x2xf32>

// LOAD-GATHER-LABEL:  @load_8D_vector(
// LOAD-GATHER-SAME:   %[[SRC:.+]]: memref<2x2x2x2x2x2x2x2xf32>,
// LOAD-GATHER:        %[[CST:.+]] = arith.constant dense<true> : vector<2x2x2x2x2x2x2x2xi1>
// LOAD-GATHER-COUNT8: vector.step
// LOAD-GATHER-COUNT7: vector.shape_cast
// LOAD-GATHER-COUNT8: vector.broadcast {{.*}} : vector<2x2x2x2x2x2x2x2xindex>
// LOAD-GATHER-COUNT7: arith.addi {{.*}} : vector<2x2x2x2x2x2x2x2xindex>
// LOAD-GATHER:        %[[SPLAT:.+]] = vector.broadcast {{.*}} : index to vector<2x2x2x2x2x2x2x2xindex>
// LOAD-GATHER:        %[[IDX:.+]] = arith.addi %[[SPLAT]], {{.*}} : vector<2x2x2x2x2x2x2x2xindex>
// LOAD-GATHER:        %[[COLLAPSE:.+]] = memref.extract_aligned_pointer_as_index %[[SRC]] : memref<2x2x2x2x2x2x2x2xf32> -> index
// LOAD-GATHER:        %[[COLLAPSE_I:.+]] = arith.index_cast %[[COLLAPSE]] : index to i64
// LOAD-GATHER:        %[[VEC:.+]] = xegpu.load %[[COLLAPSE_I]][%[[IDX]]], %[[CST]] : i64, vector<2x2x2x2x2x2x2x2xindex>, vector<2x2x2x2x2x2x2x2xi1> -> vector<2x2x2x2x2x2x2x2xf32>

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
// LOAD-ND:        %[[LOAD:.*]] = xegpu.load_nd
// LOAD-ND:        vector.transpose %[[LOAD]], [1, 0] : vector<16x8xf16> to vector<8x16xf16>

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
// LOAD-GATHER:        %[[COLLAPSE:.*]] = memref.extract_aligned_pointer_as_index %arg0 : memref<32x64xf16> -> index
// LOAD-GATHER:        %[[COLLAPSE_I:.+]] = arith.index_cast %[[COLLAPSE]] : index to i64
// LOAD-GATHER:        %[[LOAD:.*]] = xegpu.load %[[COLLAPSE_I]][%[[IDX]]], %[[CST]] : i64, vector<8x16xindex>, vector<8x16xi1> -> vector<8x16xf16>
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

// -----
gpu.module @xevm_module {
gpu.func @load_from_subview_1D(%source: memref<4096x4096xf16>, %off1: index, %off2: index) -> vector<8xf16> {
  %c0 = arith.constant 0.0 : f16
  %subview = memref.subview %source[%off1, %off2] [256, 256] [1, 1] : memref<4096x4096xf16> to memref<256x256xf16, strided<[4096, 1], offset: ?>>
  %0 = vector.transfer_read %subview[%off2, %off2], %c0
    {in_bounds = [true]} : memref<256x256xf16, strided<[4096, 1], offset: ?>>, vector<8xf16>
  gpu.return %0 : vector<8xf16>
}

// LOAD-ND-LABEL:  @load_from_subview_1D(
// LOAD-ND-SAME:   %[[SRC:.+]]: memref<4096x4096xf16>,
// LOAD-ND-SAME:   %[[OFF1:.+]]: index, %[[OFF2:.+]]: index
// LOAD-ND:        %[[ELEM_BYTES:.+]] = arith.constant 2 : index
// LOAD-ND:        %[[SUBVIEW:.+]] = memref.subview %[[SRC]][%[[OFF1]], %[[OFF2]]] [256, 256] [1, 1] : memref<4096x4096xf16> to memref<256x256xf16, strided<[4096, 1], offset: ?>>
// LOAD-ND:        %[[COLLAPSED:.+]] = memref.subview %[[SUBVIEW]][%[[OFF2]], 0]
// LOAD-ND:        %[[BASE_BUFFER:.*]], %[[OFFSET:.*]], %[[SIZES:.*]], %[[STRIDES:.*]] = memref.extract_strided_metadata %[[COLLAPSED]]
// LOAD-ND:        %[[INTPTR:.*]] = memref.extract_aligned_pointer_as_index %[[BASE_BUFFER]]
// LOAD-ND:        %[[MUL:.+]] = arith.muli %[[OFFSET]], %[[ELEM_BYTES]] : index
// LOAD-ND:        %[[ADD:.+]] = arith.addi %[[INTPTR]], %[[MUL]] : index
// LOAD-ND:        %[[I64PTR:.*]] = arith.index_cast %[[ADD]] : index to i64
// LOAD-ND:        %[[DESC:.*]] = xegpu.create_nd_tdesc %[[I64PTR]], shape : [256], strides : [1] : i64 ->
// LOAD-ND-SAME:                    !xegpu.tensor_desc<8xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
// LOAD-ND:        %[[VEC:.+]] = xegpu.load_nd %[[DESC]][%[[OFF2]]] : !xegpu.tensor_desc<8xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<8xf16>

// LOAD-GATHER-LABEL:  @load_from_subview_1D(
// LOAD-GATHER-SAME:   %[[SRC:.+]]: memref<4096x4096xf16>,
// LOAD-GATHER-SAME:   %[[OFF1:.+]]: index, %[[OFF2:.+]]: index
// LOAD-GATHER:        %[[CST:.+]] = arith.constant dense<true> : vector<8xi1>
// LOAD-GATHER:        %[[SUBVIEW:.+]] = memref.subview %[[SRC]][%[[OFF1]], %[[OFF2]]] [256, 256] [1, 1] : memref<4096x4096xf16> to memref<256x256xf16, strided<[4096, 1], offset: ?>>
// LOAD-GATHER:        %[[BB:.+]], %[[OFFSET:.+]],{{.*}},{{.*}} = memref.extract_strided_metadata %[[SUBVIEW]] : memref<256x256xf16, strided<[4096, 1], offset: ?>> -> memref<f16>, index, index, index, index, index
// LOAD-GATHER:        %[[STEP:.+]] = vector.step : vector<8xindex>
// LOAD-GATHER:        arith.muli {{.*}} : index
// LOAD-GATHER:        arith.addi %[[OFFSET]]{{.*}} : index
// LOAD-GATHER:        arith.addi {{.*}} : index
// LOAD-GATHER:        %[[SPLAT:.+]] = vector.broadcast {{.*}}:  index to vector<8xindex>
// LOAD-GATHER:        %[[IDX:.+]] = arith.addi %[[SPLAT]], %[[STEP]] : vector<8xindex>
// LOAD-GATHER:        %[[COLLAPSE:.+]] = memref.extract_aligned_pointer_as_index %[[SUBVIEW]] : memref<256x256xf16, strided<[4096, 1], offset: ?>> -> index
// LOAD-GATHER:        %[[COLLAPSE_I:.+]] = arith.index_cast %[[COLLAPSE]] : index to i64
// LOAD-GATHER:        %[[VEC:.+]] = xegpu.load %[[COLLAPSE_I]]{{\[}}%[[IDX]]{{\]}}, %[[CST]] : i64, vector<8xindex>, vector<8xi1> -> vector<8xf16>
}

// -----
gpu.module @xevm_module {
gpu.func @load_from_subview_2D(%source: memref<4096x4096xf16>, %off1: index, %off2: index) -> vector<8x16xf16> {
  %c0 = arith.constant 0.0 : f16
  %subview = memref.subview %source[%off1, %off2] [256, 256] [1, 1] : memref<4096x4096xf16> to memref<256x256xf16, strided<[4096, 1], offset: ?>>
  %0 = vector.transfer_read %subview[%off2, %off2], %c0
    {in_bounds = [true, true]} : memref<256x256xf16, strided<[4096, 1], offset: ?>>, vector<8x16xf16>
  gpu.return %0 : vector<8x16xf16>
}

// LOAD-ND-LABEL:  @load_from_subview_2D(
// LOAD-ND-SAME:   %[[SRC:.+]]: memref<4096x4096xf16>,
// LOAD-ND-SAME:   %[[OFF1:.+]]: index, %[[OFF2:.+]]: index
// LOAD-ND:        %[[ELEM_BYTES:.+]] = arith.constant 2 : index
// LOAD-ND:        %[[SUBVIEW:.+]] = memref.subview %[[SRC]][%[[OFF1]], %[[OFF2]]] [256, 256] [1, 1] : memref<4096x4096xf16> to memref<256x256xf16, strided<[4096, 1], offset: ?>>
// LOAD-ND:        %[[BASE_BUFFER:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:2, %[[STRIDES:.*]]:2 = memref.extract_strided_metadata %[[SUBVIEW]]
// LOAD-ND:        %[[INTPTR:.*]] = memref.extract_aligned_pointer_as_index %[[BASE_BUFFER]]
// LOAD-ND:        %[[MUL:.*]] = arith.muli %[[OFFSET]], %[[ELEM_BYTES]] : index
// LOAD-ND:        %[[ADD:.*]] = arith.addi %[[INTPTR]], %[[MUL]] : index
// LOAD-ND:        %[[I64PTR:.*]] = arith.index_cast %[[ADD]] : index to i64
// LOAD-ND:        %[[DESC:.*]] = xegpu.create_nd_tdesc %[[I64PTR]], shape : [256, 256], strides : [4096, 1] :
// LOAD-ND-SAME:                    i64 -> !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
// LOAD-ND:        %[[VEC:.+]] = xegpu.load_nd %[[DESC]][%[[OFF2]], %[[OFF2]]]{{.*}}-> vector<8x16xf16>
// LOAD-ND:        return %[[VEC]]

// LOAD-GATHER-LABEL:  @load_from_subview_2D(
// LOAD-GATHER-SAME:   %[[SRC:.+]]: memref<4096x4096xf16>,
// LOAD-GATHER-SAME:   %[[OFF1:.+]]: index, %[[OFF2:.+]]: index
// LOAD-GATHER:        %[[CST:.+]] = arith.constant dense<true> : vector<8x16xi1>
// LOAD-GATHER:        %[[SUBVIEW:.+]] = memref.subview %[[SRC]][%[[OFF1]], %[[OFF2]]] [256, 256] [1, 1] : memref<4096x4096xf16> to memref<256x256xf16, strided<[4096, 1], offset: ?>>
// LOAD-GATHER:        %[[BB:.+]], %[[OFFSET:.+]],{{.*}},{{.*}} = memref.extract_strided_metadata %[[SUBVIEW]] : memref<256x256xf16, strided<[4096, 1], offset: ?>> -> memref<f16>, index, index, index, index, index
// LOAD-GATHER-COUNT2: vector.step
// LOAD-GATHER-COUNT2: vector.shape_cast
// LOAD-GATHER-COUNT2: vector.broadcast
// LOAD-GATHER-COUNT2: arith.muli {{.*}} : index
// LOAD-GATHER-COUNT2: arith.addi {{.*}} : index
// LOAD-GATHER:        %[[SPLAT:.+]] = vector.broadcast {{.*}}:  index to vector<8x16xindex>
// LOAD-GATHER:        %[[IDX:.+]] = arith.addi %[[SPLAT]], {{.*}} : vector<8x16xindex>
// LOAD-GATHER:        %[[COLLAPSE:.+]] = memref.extract_aligned_pointer_as_index %[[SUBVIEW]] : memref<256x256xf16, strided<[4096, 1], offset: ?>> -> index
// LOAD-GATHER:        %[[COLLAPSE_I:.+]] = arith.index_cast %[[COLLAPSE]] : index to i64
// LOAD-GATHER:        %[[VEC:.+]] = xegpu.load %[[COLLAPSE_I]]{{\[}}%[[IDX]]{{\]}}, %[[CST]] : i64, vector<8x16xindex>, vector<8x16xi1> -> vector<8x16xf16>
}

// -----
gpu.module @xevm_module {
gpu.func @load_2D_vector_addrspace3(%source: memref<16x32xf32, 3>,
    %offset: index) -> vector<8x16xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %source[%offset, %offset], %c0
    {in_bounds = [true, true]} : memref<16x32xf32, 3>, vector<8x16xf32>
  gpu.return %0 : vector<8x16xf32>
}

// LOAD-ND-LABEL: @load_2D_vector_addrspace3
// LOAD-ND-SAME: %[[SOURCE:.+]]: memref<16x32xf32, 3>
// LOAD-ND-SAME: %[[OFFSET:.+]]: index
// LOAD-ND: %[[MEM_DESC:.+]] = xegpu.create_mem_desc %[[SOURCE]] : memref<16x32xf32, 3> -> !xegpu.mem_desc<16x32xf32>
// LOAD-ND: %[[DATA:.+]] = xegpu.load_matrix %[[MEM_DESC]][%[[OFFSET]], %[[OFFSET]]] : !xegpu.mem_desc<16x32xf32>, index, index -> vector<8x16xf32>
// LOAD-ND: gpu.return %[[DATA]] : vector<8x16xf32>

// LOAD-GATHER-LABEL: @load_2D_vector_addrspace3
// LOAD-GATHER-SAME: %[[SOURCE:.+]]: memref<16x32xf32, 3>
// LOAD-GATHER-SAME: %[[OFFSET:.+]]: index
// LOAD-GATHER: %[[MEM_DESC:.+]] = xegpu.create_mem_desc %[[SOURCE]] : memref<16x32xf32, 3> -> !xegpu.mem_desc<16x32xf32>
// LOAD-GATHER: %[[DATA:.+]] = xegpu.load_matrix %[[MEM_DESC]][%[[OFFSET]], %[[OFFSET]]] : !xegpu.mem_desc<16x32xf32>, index, index -> vector<8x16xf32>
// LOAD-GATHER: gpu.return %[[DATA]] : vector<8x16xf32>

}

// -----
gpu.module @xevm_module {
gpu.func @load_1D_vector_addrspace3(%source: memref<32xf32, 3>,
    %offset: index) -> vector<8xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %source[%offset], %c0
    {in_bounds = [true]} : memref<32xf32, 3>, vector<8xf32>
  gpu.return %0 : vector<8xf32>
}

// LOAD-ND-LABEL: @load_1D_vector_addrspace3
// LOAD-ND-SAME: %[[SOURCE:.+]]: memref<32xf32, 3>
// LOAD-ND-SAME: %[[OFFSET:.+]]: index
// LOAD-ND: %[[MEM_DESC:.+]] = xegpu.create_mem_desc %[[SOURCE]] : memref<32xf32, 3> -> !xegpu.mem_desc<32xf32>
// LOAD-ND: %[[DATA:.+]] = xegpu.load_matrix %[[MEM_DESC]][%[[OFFSET]]] : !xegpu.mem_desc<32xf32>, index -> vector<8xf32>
// LOAD-ND: gpu.return %[[DATA]] : vector<8xf32>

// LOAD-GATHER-LABEL: @load_1D_vector_addrspace3
// LOAD-GATHER-SAME: %[[SOURCE:.+]]: memref<32xf32, 3>
// LOAD-GATHER-SAME: %[[OFFSET:.+]]: index
// LOAD-GATHER: %[[MEM_DESC:.+]] = xegpu.create_mem_desc %[[SOURCE]] : memref<32xf32, 3> -> !xegpu.mem_desc<32xf32>
// LOAD-GATHER: %[[DATA:.+]] = xegpu.load_matrix %[[MEM_DESC]][%[[OFFSET]]] : !xegpu.mem_desc<32xf32>, index -> vector<8xf32>
// LOAD-GATHER: gpu.return %[[DATA]] : vector<8xf32>

}

// -----
// memref.alloca with the default address space is promoted to SLM
// (address space 3) so that the transfer_read can be lowered to
// xegpu.load_matrix.
gpu.module @xevm_module {
gpu.func @load_2D_vector_alloca_promoted_to_slm(%offset: index)
    -> vector<8x16xf32> {
  %buf = memref.alloca() : memref<16x32xf32>
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %buf[%offset, %offset], %c0
    {in_bounds = [true, true]} : memref<16x32xf32>, vector<8x16xf32>
  gpu.return %0 : vector<8x16xf32>
}

// LOAD-ND-LABEL: @load_2D_vector_alloca_promoted_to_slm
// LOAD-ND: %[[BUF:.+]] = memref.alloca() : memref<16x32xf32, 3>
// LOAD-ND: %[[MEM_DESC:.+]] = xegpu.create_mem_desc %[[BUF]] : memref<16x32xf32, 3> -> !xegpu.mem_desc<16x32xf32>
// LOAD-ND: xegpu.load_matrix %[[MEM_DESC]]

// LOAD-GATHER-LABEL: @load_2D_vector_alloca_promoted_to_slm
// LOAD-GATHER: %[[BUF:.+]] = memref.alloca() : memref<16x32xf32, 3>
// LOAD-GATHER: %[[MEM_DESC:.+]] = xegpu.create_mem_desc %[[BUF]] : memref<16x32xf32, 3> -> !xegpu.mem_desc<16x32xf32>
// LOAD-GATHER: xegpu.load_matrix %[[MEM_DESC]]

}

// -----
// memref.alloca is unconditionally promoted to SLM (address space 3) and,
// lowered to xegpu.load_matrix.
gpu.module @xevm_module {
gpu.func @load_1D_vector_alloca_promoted_to_slm(%offset: index)
    -> vector<8xf32> {
  %buf = memref.alloca() : memref<16xf32>
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %buf[%offset], %c0
    {in_bounds = [true]} : memref<16xf32>, vector<8xf32>
  gpu.return %0 : vector<8xf32>
}

// LOAD-ND-LABEL: @load_1D_vector_alloca_promoted_to_slm
// LOAD-ND: %[[BUF:.+]] = memref.alloca() : memref<16xf32, 3>
// LOAD-ND: %[[MEM_DESC:.+]] = xegpu.create_mem_desc %[[BUF]] : memref<16xf32, 3> -> !xegpu.mem_desc<16xf32>
// LOAD-ND: xegpu.load_matrix %[[MEM_DESC]]

// LOAD-GATHER-LABEL: @load_1D_vector_alloca_promoted_to_slm
// LOAD-GATHER: %[[BUF:.+]] = memref.alloca() : memref<16xf32, 3>
// LOAD-GATHER: %[[MEM_DESC:.+]] = xegpu.create_mem_desc %[[BUF]] : memref<16xf32, 3> -> !xegpu.mem_desc<16xf32>
// LOAD-GATHER: xegpu.load_matrix %[[MEM_DESC]]

}

// -----
gpu.module @xevm_module {
gpu.func @load_0D_memref_unsupported(%source: memref<f16>) -> vector<f16> {
  %c0 = arith.constant 0.0 : f16
  %0 = vector.transfer_read %source[], %c0 : memref<f16>, vector<f16>
  gpu.return %0 : vector<f16>
}

// CHECK-LABEL: @load_0D_memref_unsupported
// CHECK: vector.transfer_read

}

// -----
gpu.module @xevm_module {
gpu.func @load_0D_vector_unsupported(%source: memref<3xf32>,
    %offset: index) -> vector<f32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %source[%offset], %c0
    : memref<3xf32>, vector<f32>
  gpu.return %0 : vector<f32>
}

// LOAD-ND-LABEL: @load_0D_vector_unsupported
// LOAD-ND: vector.transfer_read

// LOAD-GATHER-LABEL: @load_0D_vector_unsupported
// LOAD-GATHER: vector.transfer_read

}

// -----
gpu.module @xevm_module {
gpu.func @transpose_1x1024x24x64(
    %arg0: memref<1x1024x24x64xf16>,
    %arg1: memref<1x24x1024x64xf16>) kernel
    attributes {known_block_size = array<i32: 256, 1, 1>} {
  %pad = ub.poison : f16
  %c0 = arith.constant 0 : index
  %block_id_x = gpu.block_id x
  %block_id_y = gpu.block_id y
  %block_id_z = gpu.block_id z
  %seq_off = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%block_id_y]
  %hid_off = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%block_id_z]
  %vec = vector.transfer_read %arg0[%c0, %seq_off, %block_id_x, %hid_off], %pad
    {in_bounds = [true, true, true, true]}
    : memref<1x1024x24x64xf16>, vector<1x16x1x8xf16>
  %transposed = vector.transpose %vec, [0, 2, 1, 3]
    : vector<1x16x1x8xf16> to vector<1x1x16x8xf16>
  vector.transfer_write %transposed, %arg1[%c0, %block_id_x, %seq_off, %hid_off]
    {in_bounds = [true, true, true, true]}
    : vector<1x1x16x8xf16>, memref<1x24x1024x64xf16>
  gpu.return
}

// The vector.transpose is folded into the transfer_read via
// CombineTransferReadOpTranspose, giving the read a mid-vector permutation map
// (d0, d1, d2, d3) -> (d0, d2, d1, d3). An nd block load can only realize an
// innermost-two-dims transpose, so the read falls back to the scattered path
// while the identity-map write still lowers to store_nd.
// LOAD-ND-LABEL: @transpose_1x1024x24x64
// LOAD-ND-DAG: %[[C1536:.+]] = arith.constant 1536 : index
// LOAD-ND-DAG: %[[C64:.+]] = arith.constant 64 : index
// LOAD-ND:     arith.muli %{{.+}}, %[[C1536]] : index
// LOAD-ND:     arith.muli %block_id_x, %[[C64]] : index
// LOAD-ND:     %[[VEC:.+]] = xegpu.load {{.*}} -> vector<1x1x16x8xf16>
// LOAD-ND:     %[[WDESC:.+]] = xegpu.create_nd_tdesc %arg1 : memref<1x24x1024x64xf16>
// LOAD-ND-SAME:  -> !xegpu.tensor_desc<1x1x16x8xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
// LOAD-ND:     xegpu.store_nd %[[VEC]], %[[WDESC]]

// LOAD-GATHER-LABEL: @transpose_1x1024x24x64
// LOAD-GATHER-DAG: %[[C1536:.+]] = arith.constant 1536 : index
// LOAD-GATHER-DAG: %[[C64:.+]] = arith.constant 64 : index
// LOAD-GATHER-DAG: %[[C65536:.+]] = arith.constant 65536 : index

// Read from memref<1x1024x24x64xf16>, strides [1572864, 1536, 64, 1].
// Scalar base offset: seq_off * 1536 (original dim1 stride),
//                    block_id_x * 64  (original dim2 stride).
// LOAD-GATHER:     arith.muli %{{.+}}, %[[C1536]] : index
// LOAD-GATHER:     arith.muli %block_id_x, %[[C64]] : index
// LOAD-GATHER:     xegpu.load {{.*}} -> vector<1x1x16x8xf16>

// Write to memref<1x24x1024x64xf16>, strides [1572864, 65536, 64, 1].
// Scalar base offset: block_id_x * 65536 (original dim1 stride),
//                    seq_off * 64        (original dim2 stride).
// LOAD-GATHER:     arith.muli %block_id_x, %[[C65536]] : index
// LOAD-GATHER:     arith.muli %{{.+}}, %[[C64]] : index
// LOAD-GATHER:     xegpu.store {{.*}}

}
