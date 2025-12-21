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
  // CHECK: xegpu.convert_layout
  // CHECK-SAME: <{input_layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>, target_layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}> : vector<32x64xf16>
  %2 = xegpu.convert_layout %a <{input_layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>,
                                target_layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}> : vector<32x64xf16>
  gpu.return
}

gpu.func @convert_layout_wg(%a: vector<32x64xf16>) {
  // CHECK: xegpu.convert_layout
  // CHECK-SAME: <{input_layout = #xegpu.layout<sg_layout = [2, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>, target_layout = #xegpu.layout<sg_layout = [4, 2], sg_data = [8, 32], lane_layout = [1, 16], lane_data = [1, 1]>}> : vector<32x64xf16>
  %2 = xegpu.convert_layout %a <{input_layout = #xegpu.layout<sg_layout = [2, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>,
                                target_layout = #xegpu.layout<sg_layout = [4, 2], sg_data = [8, 32], lane_layout = [1, 16], lane_data = [1, 1]>}> : vector<32x64xf16>
  gpu.return
}

gpu.func @slice_attr() {
  //CHECK: arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [16, 1, 1], sg_data = [1, 8, 2]>, dims = [2]>} dense<8> : vector<16x8xindex>
  %cst = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [16, 1, 1], sg_data = [1, 8, 2]>, dims = [2]>} dense<8> : vector<16x8xindex>
  gpu.return
}

gpu.func @nested_slice_attr() {
  //CHECK: arith.constant {layout_result_0 = #xegpu.slice<#xegpu.slice<#xegpu.layout<sg_layout = [16, 1, 1], sg_data = [1, 8, 2]>, dims = [2]>, dims = [1]>} dense<8> : vector<16xindex>
  %cst = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.slice<#xegpu.layout<sg_layout = [16, 1, 1], sg_data = [1, 8, 2]>, dims = [2]>, dims = [1]>} dense<8> : vector<16xindex>
  gpu.return
}

gpu.func @softmax_dim_0(%arg0: vector<256x128xf32>) -> vector<256x128xf32> {
  %cst = arith.constant dense<0.000000e+00> : vector<128xf32>
  %0 = math.exp %arg0 {layout_result_0 = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32]>} : vector<256x128xf32>
  //CHECK: vector.multi_reduction <add>, {{.*}} {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32]>, dims = [0]>} [0] : vector<256x128xf32> to vector<128xf32>
  %1 = vector.multi_reduction <add>, %0, %cst {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32]>, dims = [0]>} [0] : vector<256x128xf32> to vector<128xf32>
  //CHECK: vector.broadcast {{.*}} {layout_result_0 = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32]>} : vector<128xf32> to vector<256x128xf32>
  %2 = vector.broadcast %1 {layout_result_0 = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32]>} : vector<128xf32> to vector<256x128xf32>
  %3 = arith.divf %0, %2 {layout_result_0 = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32]>} : vector<256x128xf32>
  gpu.return %3 : vector<256x128xf32>
}

}
