// RUN: mlir-opt %s --xevm-attach-target='module=xevm_* O=3 chip=pvc' -convert-vector-to-xegpu -split-input-file | FileCheck %s --check-prefix=LOAD_ND
// RUN: mlir-opt %s -convert-vector-to-xegpu -split-input-file | FileCheck %s --check-prefix=LOAD_GATHER

gpu.module @xevm_module {
gpu.func @load_1D_vector(%source: memref<8x16x32xf32>, %offset: index) -> vector<8xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %source[%offset, %offset, %offset], %c0
    {in_bounds = [true]} : memref<8x16x32xf32>, vector<8xf32>
  gpu.return %0 : vector<8xf32>
}

// CHECK-LABEL: LOAD_ND: @load_1D_vector(
// CHECK-SAME:  LOAD_ND: %[[SRC:.+]]: memref<8x16x32xf32>,
// CHECK-SAME:  LOAD_ND: %[[OFFSET:.+]]: index
// CHECK:       LOAD_ND: %[[DESC:.+]] = xegpu.create_nd_tdesc
// CHECK-SAME:  LOAD_ND:   %[[SRC]][%[[OFFSET]], %[[OFFSET]], %[[OFFSET]]]
// CHECK-SAME:  LOAD_ND:   memref<8x16x32xf32> -> !xegpu.tensor_desc<8xf32,
// CHECK-SAME:  LOAD_ND:   boundary_check = false
// CHECK:       LOAD_ND: %[[VEC:.+]] = xegpu.load_nd %[[DESC]]{{.*}}-> vector<8xf32>
// CHECK:       LOAD_ND: return %[[VEC]]

// CHECK-LABEL: LOAD_GATHER: @load_1D_vector(
// CHECK-SAME:  LOAD_GATHER: %[[SRC:.+]]: memref<8x16x32xf32>,
// CHECK-SAME:  LOAD_GATHER: %[[OFFSET:.+]]: index
// CHECK:       LOAD_GATHER: %[[CST:.+]] = arith.constant dense<true> : vector<8xi1>
// CHECK:       LOAD_GATHER: %[[C32:.+]] = arith.constant 32 : index
// CHECK:       LOAD_GATHER: %[[C512:.+]] = arith.constant 512 : index
// CHECK:       LOAD_GATHER: %[[STEP:.+]] = vector.step : vector<8xindex>
// CHECK:       LOAD_GATHER: %[[MUL1:.+]] = arith.muli %[[OFFSET]], %[[C512]] : index
// CHECK:       LOAD_GATHER: %[[MUL2:.+]] = arith.muli %[[OFFSET]], %[[C32]] : index
// CHECK:       LOAD_GATHER: %[[ADD1:.+]] = arith.addi %[[MUL1]], %[[MUL2]] : index
// CHECK:       LOAD_GATHER: %[[ADD2:.+]] = arith.addi %[[ADD1]], %[[OFFSET]] : index
// CHECK:       LOAD_GATHER: %[[SPLAT:.+]] = vector.splat %[[ADD2]] : vector<8xindex>
// CHECK:       LOAD_GATHER: %[[IDX:.+]] = arith.addi %[[SPLAT]], %[[STEP]] : vector<8xindex>
// CHECK:       LOAD_GATHER: %[[COLLAPSE:.+]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1, 2]{{\]}} : memref<8x16x32xf32> into memref<4096xf32>
// CHECK:       LOAD_GATHER: %[[VEC:.+]] = xegpu.load %[[COLLAPSE]]{{\[}}%[[IDX]]{{\]}}, %[[CST]] : memref<4096xf32>, vector<8xindex>, vector<8xi1> -> vector<8xf32>
// CHECK:       LOAD_GATHER: return %[[VEC]]

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

// CHECK-LABEL: LOAD_ND: @load_2D_vector(
// CHECK-SAME:  LOAD_ND: %[[SRC:.+]]: memref<8x16x32xf32>,
// CHECK-SAME:  LOAD_ND: %[[OFFSET:.+]]: index
// CHECK:       LOAD_ND: %[[DESC:.+]] = xegpu.create_nd_tdesc
// CHECK-SAME:  LOAD_ND:   %[[SRC]][%[[OFFSET]], %[[OFFSET]], %[[OFFSET]]]
// CHECK-SAME:  LOAD_ND:   memref<8x16x32xf32> -> !xegpu.tensor_desc<8x16xf32,
// CHECK-SAME:  LOAD_ND:   boundary_check = false
// CHECK:       LOAD_ND: %[[VEC:.+]] = xegpu.load_nd %[[DESC]]{{.*}}-> vector<8x16xf32>
// CHECK:       LOAD_ND: return %[[VEC]]

// CHECK-LABEL: LOAD_GATHER: @load_2D_vector(
// CHECK-SAME:  LOAD_GATHER: %[[SRC:.+]]: memref<8x16x32xf32>,
// CHECK-SAME:  LOAD_GATHER: %[[OFFSET:.+]]: index
// CHECK:       LOAD_GATHER: %[[CST:.+]] = arith.constant dense<true> : vector<8x16xi1>
// CHECK:       LOAD_GATHER: %[[C32:.+]] = arith.constant 32 : index
// CHECK:       LOAD_GATHER: %[[C512:.+]] = arith.constant 512 : index
// CHECK:       LOAD_GATHER: %[[CST_0:.+]] = arith.constant dense<32> : vector<8xindex>
// CHECK:       LOAD_GATHER: %[[STEP0:.+]] = vector.step : vector<8xindex>
// CHECK:       LOAD_GATHER: %[[STEP1:.+]] = vector.step : vector<16xindex>
// CHECK:       LOAD_GATHER: %[[MUL:.+]] = arith.muli %[[STEP0]], %[[CST_0]] : vector<8xindex>
// CHECK:       LOAD_GATHER: %[[SHAPE0:.+]] = vector.shape_cast %[[MUL]] : vector<8xindex> to vector<8x1xindex>
// CHECK:       LOAD_GATHER: %[[SHAPE1:.+]] = vector.shape_cast %[[STEP1]] : vector<16xindex> to vector<1x16xindex>
// CHECK:       LOAD_GATHER: %[[BROADCAST0:.+]] = vector.broadcast %[[SHAPE0]] : vector<8x1xindex> to vector<8x16xindex>
// CHECK:       LOAD_GATHER: %[[BROADCAST1:.+]] = vector.broadcast %[[SHAPE1]] : vector<1x16xindex> to vector<8x16xindex>
// CHECK:       LOAD_GATHER: %[[ADD_VEC:.+]] = arith.addi %[[BROADCAST0]], %[[BROADCAST1]] : vector<8x16xindex>
// CHECK:       LOAD_GATHER: %[[MUL1:.+]] = arith.muli %[[OFFSET]], %[[C512]] : index
// CHECK:       LOAD_GATHER: %[[MUL2:.+]] = arith.muli %[[OFFSET]], %[[C32]] : index
// CHECK:       LOAD_GATHER: %[[ADD1:.+]] = arith.addi %[[MUL1]], %[[MUL2]] : index
// CHECK:       LOAD_GATHER: %[[ADD2:.+]] = arith.addi %[[ADD1]], %[[OFFSET]] : index
// CHECK:       LOAD_GATHER: %[[SPLAT:.+]] = vector.splat %[[ADD2]] : vector<8x16xindex>
// CHECK:       LOAD_GATHER: %[[IDX:.+]] = arith.addi %[[SPLAT]], %[[ADD_VEC]] : vector<8x16xindex>
// CHECK:       LOAD_GATHER: %[[COLLAPSE:.+]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1, 2]{{\]}} : memref<8x16x32xf32> into memref<4096xf32>
// CHECK:       LOAD_GATHER: %[[VEC:.+]] = xegpu.load %[[COLLAPSE]]{{\[}}%[[IDX]]{{\]}}, %[[CST]] : memref<4096xf32>, vector<8x16xindex>, vector<8x16xi1> -> vector<8x16xf32>
// CHECK:       LOAD_GATHER: return %[[VEC]]
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

// CHECK-LABEL: @load_zero_pad_out_of_bounds(
// CHECK-SAME:  %[[SRC:.+]]: memref<32x64xf32>,
// CHECK-SAME:  %[[OFFSET:.+]]: index
// CHECK:       %[[DESC:.+]] = xegpu.create_nd_tdesc %[[SRC]][%[[OFFSET]], %[[OFFSET]]]
// CHECK-SAME:    memref<32x64xf32> -> !xegpu.tensor_desc<8x16xf32>
// CHECK:       %[[VEC:.+]] = xegpu.load_nd %[[DESC]]{{.*}}-> vector<8x16xf32>
// CHECK:       return %[[VEC]]
}

// -----
gpu.module @xevm_module {
gpu.func @load_transposed(%source: memref<32x64xf32>,
    %offset: index) -> vector<8x16xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %source[%offset, %offset], %c0
    {permutation_map = affine_map<(d0, d1) -> (d1, d0)>,
    in_bounds = [true, true]} : memref<32x64xf32>, vector<8x16xf32>
  gpu.return %0 : vector<8x16xf32>
}

// CHECK-LABEL: @load_transposed(
// CHECK-SAME:  %[[SRC:.+]]: memref<32x64xf32>,
// CHECK-SAME:  %[[OFFSET:.+]]: index
// CHECK:       %[[DESC:.+]] = xegpu.create_nd_tdesc %[[SRC]][%[[OFFSET]], %[[OFFSET]]]
// CHECK-SAME:    memref<32x64xf32> -> !xegpu.tensor_desc<16x8xf32
// CHECK:       %[[VEC:.+]] = xegpu.load_nd %[[DESC]] <{transpose = array<i64: 1, 0>}>
// CHECK-SAME:    -> vector<8x16xf32>
// CHECK:       return %[[VEC]]
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
// CHECK-LABEL: LOAD_ND: @load_dynamic_source(
// CHECK-SAME:  LOAD_ND: %[[SRC:.+]]: memref<?x?x?xf32>,
// CHECK-SAME:  LOAD_ND: %[[OFFSET:.+]]: index
// CHECK:       LOAD_ND: %[[C2:.+]] = arith.constant 2 : index
// CHECK:       LOAD_ND: %[[C1:.+]] = arith.constant 1 : index
// CHECK:       LOAD_ND: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   LOAD_ND: %[[DIM_0:.+]] = memref.dim %[[SRC]], %[[C0]]
// CHECK-DAG:   LOAD_ND: %[[DIM_1:.+]] = memref.dim %[[SRC]], %[[C1]]
// CHECK-DAG:   LOAD_ND: %[[DIM_2:.+]] = memref.dim %[[SRC]], %[[C2]]
// CHECK:       LOAD_ND: %[[DIM_0_STRIDE:.+]] = arith.muli %[[DIM_2]], %[[DIM_1]]
// CHECK:       LOAD_ND: %[[DESC:.+]] = xegpu.create_nd_tdesc %[[SRC]][%[[OFFSET:.+]], %[[OFFSET:.+]], %[[OFFSET:.+]]]
// CHECK:       LOAD_ND: %[[VEC:.+]] = xegpu.load_nd %[[DESC]]{{.*}}-> vector<8x16xf32>
// CHECK:       LOAD_ND: return %[[VEC]]


// CHECK-LABEL: LOAD_GATHER: @load_dynamic_source(%[[ARG0:.+]]: memref<?x?x?xf32>, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index, %[[ARG3:.+]]: index)
// CHECK:       LOAD_GATHER: %[[CST:.+]] = arith.constant dense<true> : vector<8x16xi1>
// CHECK:       LOAD_GATHER: memref.extract_strided_metadata %[[ARG0]]
// CHECK:       LOAD_GATHER: %[[STEP8:.+]] = vector.step : vector<8xindex>
// CHECK:       LOAD_GATHER: %[[STEP16:.+]] = vector.step : vector<16xindex>
// CHECK:       LOAD_GATHER: %[[BSTR1:.+]] = vector.broadcast %[[STR1:.+]] : index to vector<8xindex>
// CHECK:       LOAD_GATHER: %[[MUL8:.+]] = arith.muli %[[STEP8]], %[[BSTR1]] : vector<8xindex>
// CHECK:       LOAD_GATHER: %[[SHAPE8:.+]] = vector.shape_cast %[[MUL8]] : vector<8xindex> to vector<8x1xindex>
// CHECK:       LOAD_GATHER: %[[SHAPE16:.+]] = vector.shape_cast %[[STEP16]] : vector<16xindex> to vector<1x16xindex>
// CHECK:       LOAD_GATHER: %[[BROAD8:.+]] = vector.broadcast %[[SHAPE8]] : vector<8x1xindex> to vector<8x16xindex>
// CHECK:       LOAD_GATHER: %[[BROAD16:.+]] = vector.broadcast %[[SHAPE16]] : vector<1x16xindex> to vector<8x16xindex>
// CHECK:       LOAD_GATHER: %[[ADDVEC:.+]] = arith.addi %[[BROAD8]], %[[BROAD16]] : vector<8x16xindex>
// CHECK:       LOAD_GATHER: %[[MULI1:.+]] = arith.muli %[[ARG1]], %[[STR0:.+]] : index
// CHECK:       LOAD_GATHER: %[[MULI2:.+]] = arith.muli %[[ARG2]], %[[STR1]] : index
// CHECK:       LOAD_GATHER: %[[ADDI1:.+]] = arith.addi %[[MULI1]], %[[MULI2]] : index
// CHECK:       LOAD_GATHER: %[[ADDI2:.+]] = arith.addi %[[ADDI1]], %[[ARG3]] : index
// CHECK:       LOAD_GATHER: %[[BROADIDX:.+]] = vector.broadcast %[[ADDI2]] : index to vector<8x16xindex>
// CHECK:       LOAD_GATHER: %[[FINALIDX:.+]] = arith.addi %[[BROADIDX]], %[[ADDVEC]] : vector<8x16xindex>
// CHECK:       LOAD_GATHER: %[[COLLAPSE:.+]] = memref.collapse_shape %[[ARG0]] {{\[}}[0, 1, 2]{{\]}} : memref<?x?x?xf32> into memref<?xf32>
// CHECK:       LOAD_GATHER: %[[RES:.+]] = xegpu.load %[[COLLAPSE]][%[[FINALIDX]]], %[[CST]] : memref<?xf32>, vector<8x16xindex>, vector<8x16xi1> -> vector<8x16xf32>
// CHECK:       LOAD_GATHER: gpu.return %[[RES]] : vector<8x16xf32>
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

// CHECK-LABEL: LOAD_ND: @load_dynamic_source2(
// CHECK-DAG:   LOAD_ND: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   LOAD_ND: %[[DIM:.+]] = memref.dim %{{.*}}, %[[C0]] : memref<?x8x16xf32>
// CHECK:       LOAD_ND: %[[DESC:.+]] = xegpu.create_nd_tdesc %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}], shape : [%[[DIM]], 8, 16], strides : [128, 16, 1] : memref<?x8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
// CHECK:       LOAD_ND: %[[VEC:.+]] = xegpu.load_nd %[[DESC]] : !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<8x16xf32>
// CHECK:       LOAD_ND: return %[[VEC]] : vector<8x16xf32>

// CHECK-LABEL: LOAD_GATHER: @load_dynamic_source2(
// CHECK-DAG:   LOAD_GATHER: %[[CST:.+]] = arith.constant dense<16> : vector<8xindex>
// CHECK-DAG:   LOAD_GATHER: %[[CST_0:.+]] = arith.constant dense<true> : vector<8x16xi1>
// CHECK-DAG:   LOAD_GATHER: %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG:   LOAD_GATHER: %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:   LOAD_GATHER: %[[STEP8:.+]] = vector.step : vector<8xindex>
// CHECK-DAG:   LOAD_GATHER: %[[STEP16:.+]] = vector.step : vector<16xindex>
// CHECK-DAG:   LOAD_GATHER: %[[MUL8:.+]] = arith.muli %[[STEP8]], %[[CST]] : vector<8xindex>
// CHECK-DAG:   LOAD_GATHER: %[[SHAPE8:.+]] = vector.shape_cast %[[MUL8]] : vector<8xindex> to vector<8x1xindex>
// CHECK-DAG:   LOAD_GATHER: %[[SHAPE16:.+]] = vector.shape_cast %[[STEP16]] : vector<16xindex> to vector<1x16xindex>
// CHECK-DAG:   LOAD_GATHER: %[[BCAST8:.+]] = vector.broadcast %[[SHAPE8]] : vector<8x1xindex> to vector<8x16xindex>
// CHECK-DAG:   LOAD_GATHER: %[[BCAST16:.+]] = vector.broadcast %[[SHAPE16]] : vector<1x16xindex> to vector<8x16xindex>
// CHECK-DAG:   LOAD_GATHER: %[[ADDIDX:.+]] = arith.addi %[[BCAST8]], %[[BCAST16]] : vector<8x16xindex>
// CHECK-DAG:   LOAD_GATHER: %[[MULI1:.+]] = arith.muli %arg1, %[[C128]] : index
// CHECK-DAG:   LOAD_GATHER: %[[MULI2:.+]] = arith.muli %arg2, %[[C16]] : index
// CHECK-DAG:   LOAD_GATHER: %[[ADDI1:.+]] = arith.addi %[[MULI1]], %[[MULI2]] : index
// CHECK-DAG:   LOAD_GATHER: %[[ADDI2:.+]] = arith.addi %[[ADDI1]], %arg3 : index
// CHECK-DAG:   LOAD_GATHER: %[[BCASTIDX:.+]] = vector.broadcast %[[ADDI2]] : index to vector<8x16xindex>
// CHECK-DAG:   LOAD_GATHER: %[[OFFSETS:.+]] = arith.addi %[[BCASTIDX]], %[[ADDIDX]] : vector<8x16xindex>
// CHECK-DAG:   LOAD_GATHER: %[[COLLAPSE:.+]] = memref.collapse_shape %arg0 {{\[}}[0, 1, 2]{{\]}} : memref<?x8x16xf32> into memref<?xf32>
// CHECK:       LOAD_GATHER: %[[VEC:.+]] = xegpu.load %[[COLLAPSE]]{{\[}}%[[OFFSETS]]{{\]}}, %[[CST_0]] : memref<?xf32>, vector<8x16xindex>, vector<8x16xi1> -> vector<8x16xf32> 
// CHECK:       LOAD_GATHER: return %[[VEC]] : vector<8x16xf32>

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

// CHECK-LABEL: LOAD_ND: @load_dynamic_source3(
// CHECK:       LOAD_ND: vector.transfer_read

// CHECK-LABEL: LOAD_GATHER: @load_dynamic_source3(
// CHECK-SAME:  LOAD_GATHER: %[[SRC:.+]]: memref<?x?x?x?x?xf32>, %[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index, %[[I3:.+]]: index, %[[I4:.+]]: index
// CHECK:       LOAD_GATHER: %[[CST:.+]] = arith.constant dense<true> : vector<2x4x8x16xi1>
// CHECK:       LOAD_GATHER: %[[BASE:.+]], %[[OFFSET:.+]], %[[SIZES:.*]], %[[strides:.*]] = memref.extract_strided_metadata %[[SRC]] : memref<?x?x?x?x?xf32> -> memref<f32>, index, index, index, index, index, index, index, index, index, index, index
// CHECK:       LOAD_GATHER: %[[STEP0:.+]] = vector.step : vector<2xindex>
// CHECK:       LOAD_GATHER: %[[STEP1:.+]] = vector.step : vector<4xindex>
// CHECK:       LOAD_GATHER: %[[STEP2:.+]] = vector.step : vector<8xindex>
// CHECK:       LOAD_GATHER: %[[STEP3:.+]] = vector.step : vector<16xindex>
// CHECK:       LOAD_GATHER: %[[BROAD0:.+]] = vector.broadcast %[[strides1:.+]] : index to vector<2xindex>
// CHECK:       LOAD_GATHER: %[[MUL0:.+]] = arith.muli %[[STEP0]], %[[BROAD0]] : vector<2xindex>
// CHECK:       LOAD_GATHER: %[[BROAD1:.+]] = vector.broadcast %[[strides2:.+]] : index to vector<4xindex>
// CHECK:       LOAD_GATHER: %[[MUL1:.+]] = arith.muli %[[STEP1]], %[[BROAD1]] : vector<4xindex>
// CHECK:       LOAD_GATHER: %[[BROAD2:.+]] = vector.broadcast %[[strides3:.+]] : index to vector<8xindex>
// CHECK:       LOAD_GATHER: %[[MUL2:.+]] = arith.muli %[[STEP2]], %[[BROAD2]] : vector<8xindex>
// CHECK:       LOAD_GATHER: %[[SHAPE0:.+]] = vector.shape_cast %[[MUL0]] : vector<2xindex> to vector<2x1x1x1xindex>
// CHECK:       LOAD_GATHER: %[[SHAPE1:.+]] = vector.shape_cast %[[MUL1]] : vector<4xindex> to vector<1x4x1x1xindex>
// CHECK:       LOAD_GATHER: %[[SHAPE2:.+]] = vector.shape_cast %[[MUL2]] : vector<8xindex> to vector<1x1x8x1xindex>
// CHECK:       LOAD_GATHER: %[[SHAPE3:.+]] = vector.shape_cast %[[STEP3]] : vector<16xindex> to vector<1x1x1x16xindex>
// CHECK:       LOAD_GATHER: %[[BC0:.+]] = vector.broadcast %[[SHAPE0]] : vector<2x1x1x1xindex> to vector<2x4x8x16xindex>
// CHECK:       LOAD_GATHER: %[[BC1:.+]] = vector.broadcast %[[SHAPE1]] : vector<1x4x1x1xindex> to vector<2x4x8x16xindex>
// CHECK:       LOAD_GATHER: %[[BC2:.+]] = vector.broadcast %[[SHAPE2]] : vector<1x1x8x1xindex> to vector<2x4x8x16xindex>
// CHECK:       LOAD_GATHER: %[[BC3:.+]] = vector.broadcast %[[SHAPE3]] : vector<1x1x1x16xindex> to vector<2x4x8x16xindex>
// CHECK:       LOAD_GATHER: %[[ADD0:.+]] = arith.addi %[[BC0]], %[[BC1]] : vector<2x4x8x16xindex>
// CHECK:       LOAD_GATHER: %[[ADD1:.+]] = arith.addi %[[ADD0]], %[[BC2]] : vector<2x4x8x16xindex>
// CHECK:       LOAD_GATHER: %[[ADD2:.+]] = arith.addi %[[ADD1]], %[[BC3]] : vector<2x4x8x16xindex>
// CHECK:       LOAD_GATHER: %[[MULI0:.+]] = arith.muli %[[I0]], %[[strides0:.+]] : index
// CHECK:       LOAD_GATHER: %[[MULI1:.+]] = arith.muli %[[I1]], %[[strides1:.+]] : index
// CHECK:       LOAD_GATHER: %[[ADDI0:.+]] = arith.addi %[[MULI0]], %[[MULI1]] : index
// CHECK:       LOAD_GATHER: %[[MULI2:.+]] = arith.muli %[[I2]], %[[strides2:.+]] : index
// CHECK:       LOAD_GATHER: %[[ADDI1:.+]] = arith.addi %[[ADDI0]], %[[MULI2]] : index
// CHECK:       LOAD_GATHER: %[[MULI3:.+]] = arith.muli %[[I3]], %[[strides4:.+]] : index
// CHECK:       LOAD_GATHER: %[[ADDI2:.+]] = arith.addi %[[ADDI1]], %[[MULI3]] : index
// CHECK:       LOAD_GATHER: %[[ADDI3:.+]] = arith.addi %[[ADDI2]], %[[I4]] : index
// CHECK:       LOAD_GATHER: %[[SPLAT:.+]] = vector.broadcast %[[ADDI3]] : index to vector<2x4x8x16xindex>
// CHECK:       LOAD_GATHER: %[[IDX:.+]] = arith.addi %[[SPLAT]], %[[ADD2]] : vector<2x4x8x16xindex>
// CHECK:       LOAD_GATHER: %[[COLLAPSE:.+]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1, 2, 3, 4]{{\]}} : memref<?x?x?x?x?xf32> into memref<?xf32>
// CHECK:       LOAD_GATHER: %[[VEC:.+]] = xegpu.load %[[COLLAPSE]]{{\[}}%[[IDX]]{{\]}}, %[[CST]] : memref<?xf32>, vector<2x4x8x16xindex>, vector<2x4x8x16xi1> -> vector<2x4x8x16xf32>
// CHECK:       LOAD_GATHER: return %[[VEC]]
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

// CHECK-LABEL:   @no_load_out_of_bounds_non_zero_pad(
// CHECK-COUNT-2: vector.transfer_read
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

// CHECK-LABEL: @no_load_out_of_bounds_1D_vector(
// CHECK:       vector.transfer_read
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

// CHECK-LABEL: @no_load_masked(
// CHECK:       vector.transfer_read
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

// CHECK-LABEL: @no_load_tensor(
// CHECK:       vector.transfer_read
}

// -----
gpu.module @xevm_module {
gpu.func @no_load_high_dim_vector(%source: memref<16x32x64xf32>,
    %offset: index, %arg2: index) -> vector<8x16x32xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %source[%offset, %arg2, %offset], %c0
    {in_bounds = [true, true, true]} : memref<16x32x64xf32>, vector<8x16x32xf32>
  gpu.return %0 : vector<8x16x32xf32>
}

// CHECK-LABEL: @no_load_high_dim_vector(
// CHECK:       vector.transfer_read
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

// CHECK-LABEL: @no_load_non_unit_inner_stride(
// CHECK:       vector.transfer_read
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

// CHECK-LABEL: @no_load_unsupported_map(
// CHECK:       vector.transfer_read
}

// -----
gpu.module @xevm_module {
gpu.func @no_load_transpose_unsupported_data_type(%source: memref<32x64xf16>,
    %offset: index) -> vector<8x16xf16> {
  %c0 = arith.constant 0.0 : f16
  %0 = vector.transfer_read %source[%offset, %offset], %c0
    {permutation_map = affine_map<(d0, d1) -> (d1, d0)>,
    in_bounds = [true, true]} : memref<32x64xf16>, vector<8x16xf16>
  gpu.return %0 : vector<8x16xf16>
}

// CHECK-LABEL: @no_load_transpose_unsupported_data_type(
// CHECK:       vector.transfer_read
}

