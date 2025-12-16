// RUN: mlir-opt --xevm-attach-target='module=xevm_* chip=pvc'  \
// RUN:   --xegpu-optimize-block-loads --split-input-file %s | FileCheck %s

// CHECK-LABEL: gpu.func @vector_reduce_2d(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<4x16xf32>, %[[ARG2:[0-9a-zA-Z]+]]: memref<256xf32>) {
// CHECK:      %[[ACC:.*]] = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0, 1]>} 1.000000e+00 : f32
// CHECK:      %[[TDESC:.*]] = xegpu.create_nd_tdesc %[[ARG0]] : memref<4x16xf32> -> !xegpu.tensor_desc<4x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
// CHECK:      %[[LOADED:.*]] = xegpu.load_nd %[[TDESC]][0, 0]  : !xegpu.tensor_desc<4x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<4x16xf32>
// CHECK:      %[[ACC_VEC:.*]] = vector.broadcast %cst {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0]>} : f32 to vector<16xf32>
// CHECK:      %[[LOADED_REDUCED:.*]] = vector.multi_reduction <add>, %[[LOADED]], %[[ACC_VEC]]
// CHECK-SAME: {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0]>} [0] : vector<4x16xf32> to vector<16xf32>
// CHECK:      %[[LOADED_REDUCED_FOR_CROSS:.*]] = vector.reduction <add>, %[[LOADED_REDUCED]]
// CHECK-SAME: {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0, 1]>} : vector<16xf32> into f32
gpu.module @xevm_test {
  gpu.func @vector_reduce_2d(%src: memref<4x16xf32>, %dst: memref<256xf32>) {
    %cst = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0, 1]>} 1.0 : f32
    %tdesc = xegpu.create_nd_tdesc %src : memref<4x16xf32>
      -> !xegpu.tensor_desc<4x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    %load =  xegpu.load_nd %tdesc[0, 0]
      : !xegpu.tensor_desc<4x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
      -> vector<4x16xf32>
    %reduce = vector.multi_reduction <add>, %load, %cst
     {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0, 1]>}
     [0, 1] : vector<4x16xf32> to f32
    %reduce_bcast = vector.broadcast %reduce
     {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0]>}
     : f32 to vector<16xf32>

    %offset = arith.constant {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>} dense<0> : vector<16xindex>
    %mask = arith.constant {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>} dense<1> : vector<16xi1>

    xegpu.store %reduce_bcast, %dst[%offset], %mask {layout = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0]>} : vector<16xf32>, memref<256xf32>, vector<16xindex>, vector<16xi1>
    gpu.return
  }
}
