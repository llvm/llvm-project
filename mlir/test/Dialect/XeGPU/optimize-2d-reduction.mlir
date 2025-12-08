// RUN: mlir-opt --xevm-attach-target='module=xevm_* chip=pvc'  \
// RUN:   --xegpu-optimize-block-loads --split-input-file %s | FileCheck %s

// CHECK-LABEL: gpu.func @vector_reduce_2d(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<4x16xf32>) {
// CHECK:      %[[ACC:.*]] = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [4, 1]>, dims = [0, 1]>} 1.000000e+00 : f32
// CHECK:      %[[TDESC:.*]] = xegpu.create_nd_tdesc %[[ARG0]] : memref<4x16xf32> -> !xegpu.tensor_desc<4x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [4, 1]>>
// CHECK:      %[[LOADED:.*]] = xegpu.load_nd %[[TDESC]][0, 0]  : !xegpu.tensor_desc<4x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [4, 1]>> -> vector<4x16xf32>
// CHECK:      %[[ACC_VEC:.*]] = vector.from_elements %[[ACC]] : vector<1xf32>
// CHECK:      %[[ACC_VEC_FOR_INTRA:.*]] = vector.broadcast %[[ACC_VEC]]
// CHECK-SAME: {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [4, 1]>, dims = [0]>} : vector<1xf32> to vector<16xf32>
// CHECK:      %[[LOADED_REDUCED:.*]] = vector.multi_reduction <add>, %[[LOADED]], %[[ACC_VEC_FOR_INTRA]]
// CHECK-SAME: {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [4, 1]>, dims = [0]>} [0] : vector<4x16xf32> to vector<16xf32>
// CHECK:      %[[LOADED_REDUCED_FOR_CROSS:.*]] = vector.shape_cast %[[LOADED_REDUCED]]
// CHECK-SAME: {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [4, 1]>, dims = [0, 1]>} : vector<16xf32> to vector<1x16xf32>
// CHECK:      %[[LOADED_REDUCED_2D:.*]] = vector.multi_reduction <add>, %[[LOADED_REDUCED_FOR_CROSS]], %[[ACC_VEC]]
// CHECK-SAME: {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [4, 1]>, dims = [0, 1]>} [1] : vector<1x16xf32> to vector<1xf32>
// CHECK:      %[[SCALAR_RES:.*]] = vector.extract %[[LOADED_REDUCED_2D]][0] : f32 from vector<1xf32>
gpu.module @xevm_test {
  gpu.func @vector_reduce_2d(%src: memref<4x16xf32>) {
    %cst = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [4, 1]>, dims = [0, 1]>} 1.0 : f32
    %tdesc = xegpu.create_nd_tdesc %src : memref<4x16xf32>
      -> !xegpu.tensor_desc<4x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [4, 1]>>
    %load =  xegpu.load_nd %tdesc[0, 0]
      : !xegpu.tensor_desc<4x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [4, 1]>>
      -> vector<4x16xf32>
    %reduce = vector.multi_reduction <add>, %load, %cst {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [4, 1]>, dims = [0, 1]>} [0, 1]
      : vector<4x16xf32> to f32
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @vector_reduce_2d(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<4x16xf32>) {
// CHECK:      %[[ACC:.*]] = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 4, 1]>, dims = [0, 1]>} dense<1.000000e+00> : vector<1xf32>
// CHECK:      %[[TDESC:.*]] = xegpu.create_nd_tdesc %[[ARG0]] : memref<4x16xf32> -> !xegpu.tensor_desc<4x16xf32>
// CHECK:      %[[LOADED:.*]] = xegpu.load_nd %[[TDESC]][0, 0] {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [4, 1]>} : !xegpu.tensor_desc<4x16xf32> -> vector<4x16xf32>
// CHECK:      %[[LOADED_LEADING_UNIT:.*]] = vector.shape_cast %[[LOADED]]
// CHECK-SAME: {layout_result_0 = #xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 4, 1]>} : vector<4x16xf32> to vector<1x4x16xf32
// CHECK:      %[[ACC_VEC_FOR_INTRA:.*]] = vector.broadcast %[[ACC]]
// CHECK-SAME: {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 4, 1]>, dims = [1]>} : vector<1xf32> to vector<1x16xf32>
// CHECK:      %[[LOADED_REDUCED:.*]] = vector.multi_reduction <add>, %[[LOADED_LEADING_UNIT]], %[[ACC_VEC_FOR_INTRA]]
// CHECK-SAME: {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 4, 1]>, dims = [1]>} [1] : vector<1x4x16xf32> to vector<1x16xf32>
// CHECK:      %[[LOADED_REDUCED_2D:.*]] = vector.multi_reduction <add>, %[[LOADED_REDUCED]], %[[ACC]]
// CHECK-SAME: {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 4, 1]>, dims = [1, 2]>} [1] : vector<1x16xf32> to vector<1xf32>
gpu.module @xevm_test {
  gpu.func @vector_reduce_2d(%src: memref<4x16xf32>) {
    %cst = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 4, 1]>, dims = [0, 1]>} dense<1.0> : vector<1xf32>
    %tdesc = xegpu.create_nd_tdesc %src : memref<4x16xf32>
      -> !xegpu.tensor_desc<4x16xf32>
    %load =  xegpu.load_nd %tdesc[0, 0]  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [4, 1]>} : !xegpu.tensor_desc<4x16xf32> -> vector<4x16xf32>
    %load_with_dim = vector.shape_cast %load {layout_result_0 = #xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 4, 1]>} : vector<4x16xf32> to vector<1x4x16xf32>
    %reduce = vector.multi_reduction <add>, %load_with_dim, %cst {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1,1, 16], lane_data = [1, 4, 1]>, dims = [1, 2]>} [1, 2]
      : vector<1x4x16xf32> to vector<1xf32>
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @vector_reduce_2d(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<4x64xf32>) {
// CHECK:      %[[ACC:.*]] = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 4]>, dims = [0, 1]>} dense<1.000000e+00> : vector<1xf32>
// CHECK:      %[[TDESC:.*]] = xegpu.create_nd_tdesc %[[ARG0]] : memref<4x64xf32> -> !xegpu.tensor_desc<1x64xf32>
// CHECK:      %[[LOADED:.*]] = xegpu.load_nd %[[TDESC]][0, 0] {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 4]>} : !xegpu.tensor_desc<1x64xf32> -> vector<1x64xf32>
// CHECK:      %[[LOADED_LEADING_UNIT:.*]] = vector.shape_cast %[[LOADED]]
// CHECK-SAME: {layout_result_0 = #xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 4]>} : vector<1x64xf32> to vector<1x1x64xf32>
// CHECK:      %[[ACC_VEC_FOR_INTRA:.*]] = vector.broadcast %[[ACC]]
// CHECK-SAME: {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 4]>, dims = [1]>} : vector<1xf32> to vector<1x64xf32>
// CHECK:      %[[LOADED_REDUCED:.*]] = vector.multi_reduction <add>, %[[LOADED_LEADING_UNIT]], %[[ACC_VEC_FOR_INTRA]]
// CHECK-SAME: {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 4]>, dims = [1]>} [1] : vector<1x1x64xf32> to vector<1x64xf32>
// CHECK:      %[[LOADED_REDUCED_2D:.*]] = vector.multi_reduction <add>, %[[LOADED_REDUCED]], %[[ACC]]
// CHECK-SAME: {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 4]>, dims = [1, 2]>} [1] : vector<1x64xf32> to vector<1xf32>
gpu.module @xevm_test {
  gpu.func @vector_reduce_2d(%src: memref<4x64xf32>) {
    %cst = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 4]>, dims = [0, 1]>} dense<1.0> : vector<1xf32>
    %tdesc = xegpu.create_nd_tdesc %src : memref<4x64xf32>
      -> !xegpu.tensor_desc<1x64xf32>
    %load =  xegpu.load_nd %tdesc[0, 0]  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 4]>} : !xegpu.tensor_desc<1x64xf32> -> vector<1x64xf32>
    %load_with_dim = vector.shape_cast %load {layout_result_0 = #xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 4]>} : vector<1x64xf32> to vector<1x1x64xf32>
    %reduce = vector.multi_reduction <add>, %load_with_dim, %cst {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 4]>, dims = [1, 2]>} [1, 2]
      : vector<1x1x64xf32> to vector<1xf32>
    gpu.return
  }
}
