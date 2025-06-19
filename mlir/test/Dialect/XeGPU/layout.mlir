// RUN: mlir-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s

// CHECK-LABEL: gpu.module @test {
gpu.module @test {
// CHECK: gpu.func @create_nd_tdesc_subgroup_1(%[[arg0:.*]]: memref<128x128xf32>) {
gpu.func @create_nd_tdesc_subgroup_1(%src: memref<128x128xf32>) {
  // CHECK: %[[REG:.*]] = xegpu.create_nd_tdesc %[[arg0]][0, 0] : memref<128x128xf32> -> !xegpu.tensor_desc<128x128xf32, #xegpu.layout<sg_layout = [4, 2], sg_data = [32, 64]>>
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<128x128xf32> -> !xegpu.tensor_desc<128x128xf32, #xegpu.layout<sg_layout = [4, 2], sg_data = [32, 64]>>
  gpu.return
}

// CHECK: gpu.func @create_nd_tdesc_subgroup_2(%[[arg0:.*]]: memref<128x128xf32>) {
gpu.func @create_nd_tdesc_subgroup_2(%src: memref<128x128xf32>) {
  // CHECK: %[[REG:.*]] = xegpu.create_nd_tdesc %[[arg0]][0, 0] : memref<128x128xf32> -> !xegpu.tensor_desc<128x128xf32, #xegpu.layout<sg_layout = [4, 2], sg_data = [32, 64], inst_data = [8, 16]>>
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<128x128xf32> -> !xegpu.tensor_desc<128x128xf32, #xegpu.layout<sg_layout = [4, 2], sg_data = [32, 64], inst_data = [8, 16]>>
  gpu.return
}

// CHECK: gpu.func @create_nd_tdesc_subgroup_3(%[[arg0:.*]]: memref<128x128xf32>) {
gpu.func @create_nd_tdesc_subgroup_3(%src: memref<128x128xf32>) {
  // CHECK: %[[REG:.*]] = xegpu.create_nd_tdesc %[[arg0]][0, 0] : memref<128x128xf32> -> !xegpu.tensor_desc<128x128xf32, #xegpu.layout<sg_layout = [4, 2], sg_data = [32, 64], inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<128x128xf32> -> !xegpu.tensor_desc<128x128xf32, #xegpu.layout<sg_layout = [4, 2], sg_data = [32, 64], inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
  gpu.return
}

// CHECK: gpu.func @create_nd_tdesc_wg_1(%[[arg0:.*]]: memref<24x32xf32>) {
gpu.func @create_nd_tdesc_wg_1(%src: memref<24x32xf32>) {
  // CHECK: %[[REG:.*]] = xegpu.create_nd_tdesc %arg0[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [3, 2], sg_data = [8, 16], lane_layout = [1, 16], lane_data = [8, 1]>>
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [3, 2], sg_data = [8, 16], lane_layout = [1, 16], lane_data = [8, 1]>>
  gpu.return
}

gpu.func @convert_layout(%a: vector<32x64xf16>) {
  %2 = xegpu.convert_layout %a {srcMap = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>,
                                resMap = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<32x64xf16>
  gpu.return
}

gpu.func @convert_layout_wg(%a: vector<32x64xf16>) {
  %2 = xegpu.convert_layout %a {srcMap = #xegpu.layout<sg_layout = [2, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>,
                                resMap = #xegpu.layout<sg_layout = [4, 2], sg_data = [8, 32], lane_layout = [1, 16], lane_data = [1, 1]>} : vector<32x64xf16>
  gpu.return
}

}
