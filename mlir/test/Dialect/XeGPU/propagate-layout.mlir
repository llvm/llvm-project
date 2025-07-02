// RUN: mlir-opt -xegpu-propagate-layout -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @dpas_f16(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<8x16xf16>, %[[ARG1:[0-9a-zA-Z]+]]: memref<16x16xf16>, %[[ARG2:[0-9a-zA-Z]+]]: memref<8x16xf32>) {
// CHECK: %[[CST:.*]] = arith.constant {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} dense<0.000000e+00> : vector<8x16xf32>
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][{{.*}}] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
// CHECK: %[[T1:.*]] = xegpu.create_nd_tdesc %[[ARG1]][{{.*}}] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>
// CHECK: %[[T2:.*]] = xegpu.load_nd %[[T0]]  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} :
// CHECK-SAME: !xegpu.tensor_desc<8x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<8x16xf16>
// CHECK: %[[T3:.*]] = xegpu.load_nd %[[T1]]  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>} :
// CHECK-SAME: !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>> -> vector<16x16xf16>
// CHECK: %[[T4:.*]] = xegpu.dpas %[[T2]], %[[T3]], %[[CST]] {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} :
// CHECK-SAME: vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
// CHECK: %[[T5:.*]] = xegpu.create_nd_tdesc %[[ARG2]][{{.*}}] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
// CHECK: xegpu.store_nd %[[T4]], %[[T5]] : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
func.func @dpas_f16(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.000000e+00> : vector<8x16xf32>
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
  %2 = xegpu.load_nd %0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %3 = xegpu.load_nd %1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %4 = xegpu.dpas %2, %3, %cst : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
  %5 = xegpu.create_nd_tdesc %arg2[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %4, %5  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
// CHECK-LABEL: func.func @dpas_i8(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: vector<8x32xi8>, %[[ARG1:[0-9a-zA-Z]+]]: vector<32x16xi8>, %[[ARG2:[0-9a-zA-Z]+]]: memref<8x16xi32>) {
// CHECK: %[[T0:.*]] = xegpu.dpas %[[ARG0]], %[[ARG1]] {layout_result_0 = #xegpu.layout<lane_layout = [1, 16],
func.func @dpas_i8(%arg0: vector<8x32xi8>, %arg1: vector<32x16xi8>, %arg2: memref<8x16xi32>) {
  %c0 = arith.constant 0 : index
  %0 = xegpu.dpas %arg0, %arg1 : vector<8x32xi8>, vector<32x16xi8> -> vector<8x16xi32>
  %1 = xegpu.create_nd_tdesc %arg2[%c0, %c0] : memref<8x16xi32> -> !xegpu.tensor_desc<8x16xi32>
  xegpu.store_nd %0, %1  : vector<8x16xi32>, !xegpu.tensor_desc<8x16xi32>
  return
}

// -----
// CHECK-LABEL: func.func @load_with_transpose_effect(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<8x16xf16>, %[[ARG0:[0-9a-zA-Z]+]]: memref<16x16xf16>, %[[ARG0:[0-9a-zA-Z]+]]: memref<8x16xf32>) {
// CHECK: %{{.*}} = xegpu.load_nd %{{.*}} <{transpose = array<i64: 1, 0>}> {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>} :
// CHECK-SAME: !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2]>> -> vector<16x16xf16>
func.func @load_with_transpose_effect(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.000000e+00> : vector<8x16xf32>
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
  %2 = xegpu.load_nd %0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %3 = xegpu.load_nd %1 <{transpose = array<i64: 1, 0>}> : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %4 = xegpu.dpas %2, %3, %cst : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
  %5 = xegpu.create_nd_tdesc %arg2[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %4, %5  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
// CHECK-LABEL: func.func @vector_transpose(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<8x16xf16>, %[[ARG1:[0-9a-zA-Z]+]]: memref<16x16xf16>, %[[ARG2:[0-9a-zA-Z]+]]: memref<8x16xf32>) {
// CHECK: %{{.*}} = vector.transpose %{{.*}}, [1, 0] {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>} : vector<16x16xf16> to vector<16x16xf16>
func.func @vector_transpose(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.000000e+00> : vector<8x16xf32>
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
  %2 = xegpu.load_nd %0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %3 = xegpu.load_nd %1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %4 = vector.transpose %3, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
  %5 = xegpu.dpas %2, %4, %cst : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
  %6 = xegpu.create_nd_tdesc %arg2[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %5, %6  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
// CHECK-LABEL: func.func @extf_truncf(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: !xegpu.tensor_desc<8x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>, %[[ARG1:[0-9a-zA-Z]+]]:
// CHECK-SAME: !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>) -> vector<8x16xf32> {
// CHECK: %[[T2:.*]] = arith.extf %{{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>} : vector<16x16xf16> to vector<16x16xf32>
// CHECK-NEXT: %{{.*}} = arith.truncf %[[T2]] {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>} : vector<16x16xf32> to vector<16x16xf16>
func.func @extf_truncf(%arg0: !xegpu.tensor_desc<8x16xf16>, %arg1: !xegpu.tensor_desc<16x16xf16>) -> vector<8x16xf32> {
  %0 = xegpu.load_nd %arg0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = arith.extf %1 : vector<16x16xf16> to vector<16x16xf32>
  %3 = arith.truncf %2 : vector<16x16xf32> to vector<16x16xf16>
  %4 = xegpu.dpas %0, %3 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %4 : vector<8x16xf32>
}

// -----
// CHECK-LABEL: func.func @load_gather_with_chunksize(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<8x16xf16>, %[[ARG1:[0-9a-zA-Z]+]]: memref<256xf16>, %[[ARG2:[0-9a-zA-Z]+]]: memref<8x16xf32>) {
// CHECK: %[[CST:.*]] = arith.constant {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>}
// CHECK-SAME:  dense<[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240]> : vector<16xindex>
// CHECK-NEXT: %[[CST0:.*]] = arith.constant {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>} dense<true> : vector<16xi1>
// CHECK-NEXT: %[[T2:.*]] = xegpu.create_tdesc %[[ARG1]], %[[CST]] : memref<256xf16>, vector<16xindex> ->
// CHECK-SAME: !xegpu.tensor_desc<16x16xf16, #xegpu.scatter_tdesc_attr<chunk_size = 16 : i64>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2]>>
// CHECK-NEXT: %{{.*}} = xegpu.load %[[T2]], %[[CST0]]  {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2]>}
// CHECK-SAME: !xegpu.tensor_desc<16x16xf16, #xegpu.scatter_tdesc_attr<chunk_size = 16 : i64>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2]>>, vector<16xi1> -> vector<16x16xf16>
func.func @load_gather_with_chunksize(%arg0: memref<8x16xf16>, %arg1: memref<256xf16>, %arg2: memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.load_nd %0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %cst = arith.constant dense<[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240]> : vector<16xindex>
  %cst_0 = arith.constant dense<true> : vector<16xi1>
  %2 = xegpu.create_tdesc %arg1, %cst : memref<256xf16>, vector<16xindex> -> !xegpu.tensor_desc<16x16xf16, #xegpu.scatter_tdesc_attr<chunk_size = 16 : i64>>
  %3 = xegpu.load %2, %cst_0 : !xegpu.tensor_desc<16x16xf16, #xegpu.scatter_tdesc_attr<chunk_size = 16 : i64>>, vector<16xi1> -> vector<16x16xf16>
  %4 = vector.transpose %3, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
  %5 = xegpu.dpas %1, %4 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  %6 = xegpu.create_nd_tdesc %arg2[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %5, %6  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
// CHECK-LABEL: func.func @load_gather_1d(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<256xf32>, %[[ARG1:[0-9a-zA-Z]+]]: !xegpu.tensor_desc<16xf32, #xegpu.layout<lane_layout = [16], lane_data = [1]>>) {
// CHECK: %[[CST:.*]] = arith.constant {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>}
// CHECK-SAME: dense<[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240]> : vector<16xindex>
// CHECK-NEXT: %[[CST0:.*]] = arith.constant {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>} dense<true> : vector<16xi1>
// CHECK-NEXT: %[[T0:.*]] = xegpu.create_tdesc %[[ARG0]], %[[CST]] : memref<256xf32>, vector<16xindex> ->
// CHECK-SAME: !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>, #xegpu.layout<lane_layout = [16], lane_data = [1]>>
// CHECK-NEXT: %{{.*}} = xegpu.load %[[T0]], %[[CST0]]  {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>} :
// CHECK-SAME: !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>, #xegpu.layout<lane_layout = [16], lane_data = [1]>>, vector<16xi1> -> vector<16xf32>
func.func @load_gather_1d(%arg0: memref<256xf32>, %arg1: !xegpu.tensor_desc<16xf32>) {
  %cst = arith.constant dense<[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240]> : vector<16xindex>
  %cst_0 = arith.constant dense<true> : vector<16xi1>
  %0 = xegpu.create_tdesc %arg0, %cst : memref<256xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
  %1 = xegpu.load %0, %cst_0 : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1> -> vector<16xf32>
  xegpu.store_nd %1, %arg1  : vector<16xf32>, !xegpu.tensor_desc<16xf32>
  return
}

// -----
// CHECK-LABEL: func.func @store_scatter_with_chunksize(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<128xf32>) {
// CHECK: %[[T0:.*]] = xegpu.create_tdesc %[[ARG0]], %{{.*}} : memref<128xf32>, vector<16xindex> ->
// CHECK-SAME: !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<chunk_size = 8 : i64>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>>
// CHECK-NEXT: xegpu.store %{{.*}}, %[[T0]], %{{.*}} : vector<16x8xf32>, !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<chunk_size = 8 : i64>,
// CHECK-SAME: #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>>, vector<16xi1>
func.func @store_scatter_with_chunksize(%arg0: memref<128xf32>) {
  %cst = arith.constant dense<1.000000e+00> : vector<16x8xf32>
  %cst_0 = arith.constant dense<true> : vector<16xi1>
  %cst_1 = arith.constant dense<[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240]> : vector<16xindex>
  %0 = xegpu.create_tdesc %arg0, %cst_1 : memref<128xf32>, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<chunk_size = 8 : i64>>
  xegpu.store %cst, %0, %cst_0 : vector<16x8xf32>, !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<chunk_size = 8 : i64>>, vector<16xi1>
  return
}

// -----
// CHECK-LABEL: func.func @store_scatter_1d(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: vector<16xf32>, %[[ARG1:[0-9a-zA-Z]+]]: memref<256xf32>) {
// CHECK: xegpu.store %[[ARG0]], %{{.*}}, %{{.*}}  : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>,
// CHECK-SAME: #xegpu.layout<lane_layout = [16], lane_data = [1]>>, vector<16xi1>
func.func @store_scatter_1d(%arg0: vector<16xf32>, %arg1: memref<256xf32>) {
  %cst = arith.constant dense<[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240]> : vector<16xindex>
  %cst_0 = arith.constant dense<true> : vector<16xi1>
  %0 = xegpu.create_tdesc %arg1, %cst : memref<256xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
  xegpu.store %arg0, %0, %cst_0  : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1>
  return
}

// -----
// CHECK-LABEL: func.func @vector_bitcast_i16_to_f16(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<8x16xi16>, %[[ARG1:[0-9a-zA-Z]+]]: memref<16x16xi16>, %[[ARG2:[0-9a-zA-Z]+]]: memref<8x16xf32>) {
// CHECK: %{{.*}} = vector.bitcast %{{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<8x16xi16> to vector<8x16xf16>
// CHECK: %{{.*}} = vector.bitcast %{{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>} : vector<16x16xi16> to vector<16x16xf16>
func.func @vector_bitcast_i16_to_f16(%arg0: memref<8x16xi16>, %arg1: memref<16x16xi16>, %arg2: memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<8x16xi16> -> !xegpu.tensor_desc<8x16xi16>
  %1 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<16x16xi16> -> !xegpu.tensor_desc<16x16xi16>
  %2 = xegpu.load_nd %0  : !xegpu.tensor_desc<8x16xi16> -> vector<8x16xi16>
  %3 = xegpu.load_nd %1  : !xegpu.tensor_desc<16x16xi16> -> vector<16x16xi16>
  %4 = vector.bitcast %2 : vector<8x16xi16> to vector<8x16xf16>
  %5 = vector.bitcast %3 : vector<16x16xi16> to vector<16x16xf16>
  %6 = xegpu.dpas %4, %5 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  %7 = xegpu.create_nd_tdesc %arg2[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %6, %7  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
// CHECK-LABEL: func.func @binary_op_one_use(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: !xegpu.tensor_desc<8x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>,
// CHECK-SAME: %[[ARG1:[0-9a-zA-Z]+]]: !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>,
// CHECK-SAME: %[[ARG2:[0-9a-zA-Z]+]]: !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>) {
// CHECK: %[[T1:.*]] = xegpu.load_nd %[[ARG1]]  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>} :
// CHECK-SAME: !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>> -> vector<16x16xf16>
// CHECK-NEXT: %[[T2:.*]] = xegpu.load_nd %[[ARG1]]  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>} :
// CHECK-SAME: !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>> -> vector<16x16xf16>
// CHECK-NEXT: %{{.*}} = arith.addf %[[T1]], %[[T2]] {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>} : vector<16x16xf16>
func.func @binary_op_one_use(%arg0: !xegpu.tensor_desc<8x16xf16>, %arg1: !xegpu.tensor_desc<16x16xf16>, %arg2: !xegpu.tensor_desc<8x16xf32>) {
  %0 = xegpu.load_nd %arg0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %3 = arith.addf %1, %2 : vector<16x16xf16>
  %4 = xegpu.dpas %0, %3 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  xegpu.store_nd %4, %arg2  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
// CHECK-LABEL: func.func @binary_op_multiple_uses(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: !xegpu.tensor_desc<8x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>,
// CHECK-SAME: %[[ARG1:[0-9a-zA-Z]+]]: !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>,
// CHECK-SAME: %[[ARG2:[0-9a-zA-Z]+]]: !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>,
// CHECK-SAME: %[[ARG3:[0-9a-zA-Z]+]]: !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>) {
// CHECK: %[[T2:.*]] = arith.addf %{{.*}}, %{{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<16x16xf16>
// CHECK: %[[T3:.*]] = xegpu.dpas %{{.*}}, %[[T2]] {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
// CHECK-NEXT: xegpu.store_nd %[[T3]], %[[ARG2]]  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
// CHECK-NEXT: xegpu.store_nd %[[T2]], %[[ARG3]]  : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
func.func @binary_op_multiple_uses(%arg0: !xegpu.tensor_desc<8x16xf16>, %arg1: !xegpu.tensor_desc<16x16xf16>, %arg2: !xegpu.tensor_desc<8x16xf32>, %arg3: !xegpu.tensor_desc<16x16xf16>) {
  %0 = xegpu.load_nd %arg0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %cst = arith.constant dense<1.000000e+00> : vector<16x16xf16>
  %2 = arith.addf %1, %cst : vector<16x16xf16>
  %3 = xegpu.dpas %0, %2 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  xegpu.store_nd %3, %arg2  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %2, %arg3  : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16>
  return
}

// -----
// CHECK-LABEL: func.func @for_op(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<8x128xf16>, %[[ARG1:[0-9a-zA-Z]+]]: memref<128x16xf16>, %[[ARG2:[0-9a-zA-Z]+]]: memref<8x16xf32>) {
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}] : memref<8x128xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
// CHECK-NEXT: %[[T1:.*]] = xegpu.create_nd_tdesc %[[ARG1]][%{{.*}}] : memref<128x16xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>
// CHECK-NEXT: %[[CST:.*]] = arith.constant {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} dense<0.000000e+00> : vector<8x16xf32>
// CHECK-NEXT: %[[T2:.*]]:3 = scf.for %{{.*}} iter_args(%[[ARG4:.*]] = %[[T0]], %[[ARG5:.*]] = %[[T1]], %[[ARG6:.*]] = %[[CST]]) ->
// CHECK-SAME: (!xegpu.tensor_desc<8x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>, !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>, vector<8x16xf32>) {
// CHECK-NEXT:   %[[T4:.*]] = xegpu.load_nd %[[ARG4]]  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} :
// CHECK-SAME: !xegpu.tensor_desc<8x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<8x16xf16>
// CHECK-NEXT:   %[[T5:.*]] = xegpu.load_nd %[[ARG5]]  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>} :
// CHECK-SAME: !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>> -> vector<16x16xf16>
// CHECK-NEXT:   %[[T6:.*]] = xegpu.dpas %[[T4]], %[[T5]], %[[ARG6]] {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} :
// CHECK-SAME: vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
// CHECK-NEXT:   %[[T7:.*]] = xegpu.update_nd_offset %[[ARG4]], [{{.*}}] : !xegpu.tensor_desc<8x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
// CHECK-NEXT:   %[[T8:.*]] = xegpu.update_nd_offset %[[ARG5]], [{{.*}}] : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>
// CHECK-NEXT:   scf.yield %[[T7]], %[[T8]], %[[T6]] : !xegpu.tensor_desc<8x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>,
// CHECK-SAME: !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>, vector<8x16xf32>
// CHECK-NEXT: } {layout_result_2 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
// CHECK-NEXT: %[[T3:.*]] = xegpu.create_nd_tdesc %[[ARG2]][{{.*}}] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
// CHECK-NEXT: xegpu.store_nd %[[T2]]#2, %[[T3]] : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
func.func @for_op(%arg0: memref<8x128xf16>, %arg1: memref<128x16xf16>, %arg2: memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c16 = arith.constant 16 : index
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<8x128xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<128x16xf16> -> !xegpu.tensor_desc<16x16xf16>
  %cst = arith.constant dense<0.000000e+00> : vector<8x16xf32>
  %2:3 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %0, %arg5 = %1, %arg6 = %cst) -> (!xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<16x16xf16>, vector<8x16xf32>) {
    %4 = xegpu.load_nd %arg4  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
    %5 = xegpu.load_nd %arg5  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    %6 = xegpu.dpas %4, %5, %arg6 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    %7 = xegpu.update_nd_offset %arg4, [%c0, %c16] : !xegpu.tensor_desc<8x16xf16>
    %8 = xegpu.update_nd_offset %arg5, [%c16, %c0] : !xegpu.tensor_desc<16x16xf16>
    scf.yield %7, %8, %6 : !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<16x16xf16>, vector<8x16xf32>
  }
  %3 = xegpu.create_nd_tdesc %arg2[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %2#2, %3  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
// CHECK-LABEL: func.func @if_single_use(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: !xegpu.tensor_desc<8x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>,
// CHECK-SAME: %[[ARG1:[0-9a-zA-Z]+]]: !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>,
// CHECK-SAME: %[[ARG2:[0-9a-zA-Z]+]]: i1, %[[ARG3:[0-9a-zA-Z]+]]: !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>) {
// CHECK:  %{{.*}} = scf.if %[[ARG2]] -> (vector<16x16xf16>) {
// CHECK-NEXT:    %[[T3:.*]] = xegpu.load_nd %[[ARG1]]  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>} :
// CHECK-SAME: !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>> -> vector<16x16xf16>
// CHECK-NEXT:    scf.yield %[[T3]] : vector<16x16xf16>
// CHECK-NEXT:  } else {
// CHECK-NEXT:    %[[T4:.*]] = xegpu.load_nd %[[ARG1]]  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>} :
// CHECK-SAME: !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>> -> vector<16x16xf16>
// CHECK-NEXT:    scf.yield %[[T4]] : vector<16x16xf16>
// CHECK-NEXT:  } {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>}
func.func @if_single_use(%arg0: !xegpu.tensor_desc<8x16xf16>, %arg1: !xegpu.tensor_desc<16x16xf16>, %arg2: i1, %arg3: !xegpu.tensor_desc<8x16xf32>) {
  %0 = xegpu.load_nd %arg0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = scf.if %arg2 -> (vector<16x16xf16>) {
    %3 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    scf.yield %3 : vector<16x16xf16>
  } else {
    %3 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    scf.yield %3 : vector<16x16xf16>
  }
  %2 = xegpu.dpas %0, %1 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  xegpu.store_nd %2, %arg3  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
// CHECK-LABEL: func.func @if_multiple_uses(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: !xegpu.tensor_desc<8x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>,
// CHECK-SAME: %[[ARG1:[0-9a-zA-Z]+]]: !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>,
// CHECK-SAME: %[[ARG2:[0-9a-zA-Z]+]]: i1, %[[ARG3:[0-9a-zA-Z]+]]: !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>,
// CHECK-SAME: %[[ARG4:[0-9a-zA-Z]+]]: !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>) {
// CHECK: %[[T1:.*]] = scf.if %[[ARG2]] -> (vector<16x16xf16>) {
// CHECK-NEXT:       %[[T3:.*]] = xegpu.load_nd %[[ARG1]]  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} :
// CHECK-SAME: !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<16x16xf16>
// CHECK-NEXT:       scf.yield %[[T3]] : vector<16x16xf16>
// CHECK-NEXT:     } else {
// CHECK-NEXT:       %[[T4:.*]] = xegpu.load_nd %[[ARG1]]  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} :
// CHECK-SAME: !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<16x16xf16>
// CHECK-NEXT:       scf.yield %[[T4]] : vector<16x16xf16>
// CHECK-NEXT:     } {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
func.func @if_multiple_uses(%arg0: !xegpu.tensor_desc<8x16xf16>, %arg1: !xegpu.tensor_desc<16x16xf16>, %arg2: i1, %arg3: !xegpu.tensor_desc<8x16xf32>, %arg4: !xegpu.tensor_desc<16x16xf16>) {
  %0 = xegpu.load_nd %arg0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = scf.if %arg2 -> (vector<16x16xf16>) {
    %3 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    scf.yield %3 : vector<16x16xf16>
  } else {
    %3 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    scf.yield %3 : vector<16x16xf16>
  }
  %2 = xegpu.dpas %0, %1 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  xegpu.store_nd %2, %arg3  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %1, %arg4  : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16>
  return
}

// -----
// CHECK-LABEL: func.func @vector_outer_reduction(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: vector<16x16xf32>, %[[ARG1:[0-9a-zA-Z]+]]: !xegpu.tensor_desc<16xf32, #xegpu.layout<lane_layout = [16], lane_data = [1]>>) {
// CHECK: %{{.*}} = vector.multi_reduction <add>, %[[ARG0]], %{{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>} [0] : vector<16x16xf32> to vector<16xf32>
func.func @vector_outer_reduction(%arg0: vector<16x16xf32>, %arg1: !xegpu.tensor_desc<16xf32>) {
  %cst = arith.constant dense<0.000000e+00> : vector<16xf32>
  %0 = vector.multi_reduction <add>, %arg0, %cst [0] : vector<16x16xf32> to vector<16xf32>
  xegpu.store_nd %0, %arg1  : vector<16xf32>, !xegpu.tensor_desc<16xf32>
  return
}

// -----
// CHECK-LABEL: func.func @vector_inner_reduction(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: vector<16x16xf32>, %[[ARG1:[0-9a-zA-Z]+]]: !xegpu.tensor_desc<16xf32, #xegpu.layout<lane_layout = [16], lane_data = [1]>>) {
// CHECK: %{{.*}} = vector.multi_reduction <add>, %[[ARG0]], %{{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>} [1] : vector<16x16xf32> to vector<16xf32>
func.func @vector_inner_reduction(%arg0: vector<16x16xf32>, %arg1: !xegpu.tensor_desc<16xf32>) {
  %cst = arith.constant dense<0.000000e+00> : vector<16xf32>
  %0 = vector.multi_reduction <add>, %arg0, %cst [1] : vector<16x16xf32> to vector<16xf32>
  xegpu.store_nd %0, %arg1  : vector<16xf32>, !xegpu.tensor_desc<16xf32>
  return
}

// -----
// CHECK-LABEL: func.func @update_nd_offset_1d(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<256xf32>) {
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}] : memref<256xf32> -> !xegpu.tensor_desc<16xf32, #xegpu.layout<lane_layout = [16], lane_data = [1]>>
// CHECK-NEXT: %[[T1:.*]] = xegpu.update_nd_offset %[[T0]], [%{{.*}}] : !xegpu.tensor_desc<16xf32, #xegpu.layout<lane_layout = [16], lane_data = [1]>>
func.func @update_nd_offset_1d(%arg0: memref<256xf32>){
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %1 = arith.constant dense<1.000000e+00> : vector<16xf32>
  %0 = xegpu.create_nd_tdesc %arg0[%c0] : memref<256xf32> -> !xegpu.tensor_desc<16xf32>
  %2 = xegpu.update_nd_offset %0, [%c32] : !xegpu.tensor_desc<16xf32>
  xegpu.store_nd %1, %2 : vector<16xf32>, !xegpu.tensor_desc<16xf32>
  return
}

// -----
// CHECK-LABEL: func.func @update_nd_offset_2d(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<256x256xf32>) {
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}, %{{.*}}] : memref<256x256xf32> -> !xegpu.tensor_desc<16x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
// CHECK-NEXT: %[[T1:.*]] = xegpu.update_nd_offset %[[T0]], [%{{.*}}, %{{.*}}] : !xegpu.tensor_desc<16x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
func.func @update_nd_offset_2d(%arg0: memref<256x256xf32>){
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %1 = arith.constant dense<1.000000e+00> : vector<16x16xf32>
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<256x256xf32> -> !xegpu.tensor_desc<16x16xf32>
  %2 = xegpu.update_nd_offset %0, [%c32, %c32] : !xegpu.tensor_desc<16x16xf32>
  xegpu.store_nd %1, %2 : vector<16x16xf32>, !xegpu.tensor_desc<16x16xf32>
  return
}

// -----
// CHECK-LABEL: func.func @prefetch_2d(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<256x256xf16>) {
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}, %{{.*}}] : memref<256x256xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
// CHECK-NEXT: xegpu.prefetch_nd %[[T0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
func.func @prefetch_2d(%arg0: memref<256x256xf16>){
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<256x256xf16> -> !xegpu.tensor_desc<16x16xf16>
  xegpu.prefetch_nd %0 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>: !xegpu.tensor_desc<16x16xf16>
  return
}

// -----
// CHECK-LABEL: func.func @prefetch_1d(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<256xf16>) {
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}] : memref<256xf16> -> !xegpu.tensor_desc<16xf16, #xegpu.layout<lane_layout = [16], lane_data = [1]>>
// CHECK-NEXT: xegpu.prefetch_nd %[[T0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<16xf16, #xegpu.layout<lane_layout = [16], lane_data = [1]>>
func.func @prefetch_1d(%arg0: memref<256xf16>){
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0[%c0] : memref<256xf16> -> !xegpu.tensor_desc<16xf16>
  xegpu.prefetch_nd %0 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>: !xegpu.tensor_desc<16xf16>
  return
}

// -----
// CHECK-LABEL: func.func @test_scf_while_and_condition(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<256xf32>, %[[ARG1:[0-9a-zA-Z]+]]: memref<256xf32>) {
// CHECK: %{{.*}}:3 = scf.while ({{.*}}) : (vector<16xf32>, i32, !xegpu.tensor_desc<16xf32, #xegpu.layout<lane_layout = [16], lane_data = [1]>>)
// CHECK-SAME: -> (vector<16xf32>, i32, !xegpu.tensor_desc<16xf32, #xegpu.layout<lane_layout = [16], lane_data = [1]>>) {
// CHECK:       scf.condition(%{{.*}}) {{.*}} : vector<16xf32>, i32, !xegpu.tensor_desc<16xf32, #xegpu.layout<lane_layout = [16], lane_data = [1]>>
// CHECK-NEXT: } do {
// CHECK-NEXT: ^bb0(%{{.*}}: vector<16xf32>, %{{.*}}: i32, %{{.*}}: !xegpu.tensor_desc<16xf32, #xegpu.layout<lane_layout = [16], lane_data = [1]>>):
// CHECK:     scf.yield {{.*}} : vector<16xf32>, i32, !xegpu.tensor_desc<16xf32, #xegpu.layout<lane_layout = [16], lane_data = [1]>>
// CHECK-NEXT: } attributes {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>}
func.func @test_scf_while_and_condition(%arg0: memref<256xf32>, %arg1: memref<256xf32>) {
  %c0 = arith.constant 0 : i32
  %c16 = arith.constant 16 : i32
  %c256 = arith.constant 256 : i32
  %0 = xegpu.create_nd_tdesc %arg0[0] : memref<256xf32> -> !xegpu.tensor_desc<16xf32>
  %1 = xegpu.load_nd %0  : !xegpu.tensor_desc<16xf32> -> vector<16xf32>
  %2 = xegpu.create_nd_tdesc %arg1[0] : memref<256xf32> -> !xegpu.tensor_desc<16xf32>

  %3:3 = scf.while (%arg2 = %1, %arg3 = %c0, %arg4 = %0) : (vector<16xf32>, i32, !xegpu.tensor_desc<16xf32>)
    -> (vector<16xf32>, i32, !xegpu.tensor_desc<16xf32>) {
    %4 = arith.cmpi slt, %arg3, %c256 : i32
    scf.condition(%4) %arg2, %arg3, %arg4 : vector<16xf32>, i32, !xegpu.tensor_desc<16xf32>
  } do {
  ^bb0(%arg2: vector<16xf32>, %arg3: i32, %arg4: !xegpu.tensor_desc<16xf32>):
    xegpu.store_nd %arg2, %2  : vector<16xf32>, !xegpu.tensor_desc<16xf32>
    %4 = arith.addi %arg3, %c16 : i32
    %5 = xegpu.update_nd_offset %arg4, [16] : !xegpu.tensor_desc<16xf32>
    %6 = xegpu.load_nd %5  : !xegpu.tensor_desc<16xf32> -> vector<16xf32>
    scf.yield %6, %4, %5 : vector<16xf32>, i32, !xegpu.tensor_desc<16xf32>
  }
  return
}
