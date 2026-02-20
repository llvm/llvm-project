// RUN: mlir-opt --allow-unregistered-dialect --xevm-attach-target='module=xevm_* chip=pvc' \
// RUN: --xegpu-sg-to-wi-distribute-experimental --split-input-file %s --canonicalize --cse | FileCheck %s

// CHECK-LABEL: gpu.func @gemm(%arg0: memref<1024x1024xbf16>, %arg1: memref<1024x1024xbf16>, %arg2: memref<1024x1024xf32>)
// CHECK-SAME: attributes {intel_reqd_sub_group_size = 16 : i32}
// CHECK-DAG  : %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG  : %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG  : %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG  : %[[C1024:.*]] = arith.constant 1024 : index
// CHECK-DAG  : %[[BID_X:.*]] = gpu.block_id  x
// CHECK-DAG  : %[[BID_Y:.*]] = gpu.block_id  y
// CHECK-DAG  : %[[MUL_X:.*]] = arith.muli %[[BID_X]], %[[C8]] : index
// CHECK-DAG  : %[[MUL_Y:.*]] = arith.muli %[[BID_Y]], %[[C16]] : index
// CHECK      : %[[TD_C:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
// CHECK      : %[[LOAD_C:.*]] = xegpu.load_nd %[[TD_C]][%[[MUL_X]], %[[MUL_Y]]] : !xegpu.tensor_desc<8x16xf32> -> vector<8xf32>
// CHECK-DAG  : %[[CAST_C:.*]] = vector.shape_cast %[[LOAD_C]] : vector<8xf32> to vector<8x1xf32>
// CHECK-DAG  : %[[TD_A:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<1024x1024xbf16> -> !xegpu.tensor_desc<8x16xbf16>
// CHECK-DAG  : %[[TD_B:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<1024x1024xbf16> -> !xegpu.tensor_desc<16x16xbf16>
// CHECK      : %[[FOR:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C1024]] step %[[C16]] iter_args(%[[ACC:.*]] = %[[CAST_C]]) -> (vector<8x1xf32>) {
// CHECK-DAG  :   %[[LOAD_A:.*]] = xegpu.load_nd %[[TD_A]][%[[MUL_X]], %[[IV]]] : !xegpu.tensor_desc<8x16xbf16> -> vector<8xbf16>
// CHECK-DAG  :   %[[LOAD_B:.*]] = xegpu.load_nd %[[TD_B]][%[[IV]], %[[MUL_Y]]] <{packed}> : !xegpu.tensor_desc<16x16xbf16> -> vector<16xbf16>
// CHECK-DAG  :   %[[CAST_ACC:.*]] = vector.shape_cast %[[ACC]] : vector<8x1xf32> to vector<8xf32>
// CHECK      :   %[[DPAS:.*]] = xegpu.dpas %[[LOAD_A]], %[[LOAD_B]], %[[CAST_ACC]] : vector<8xbf16>, vector<16xbf16>, vector<8xf32> -> vector<8xf32>
// CHECK      :   %[[CAST_DPAS:.*]] = vector.shape_cast %[[DPAS]] : vector<8xf32> to vector<8x1xf32>
// CHECK      :   scf.yield %[[CAST_DPAS]] : vector<8x1xf32>
// CHECK      : } {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
// CHECK      : %[[CAST_FOR:.*]] = vector.shape_cast %[[FOR]] : vector<8x1xf32> to vector<8xf32>
// CHECK      : xegpu.store_nd %[[CAST_FOR]], %[[TD_C]][%[[MUL_X]], %[[MUL_Y]]] : vector<8xf32>, !xegpu.tensor_desc<8x16xf32>
// CHECK      : gpu.return
gpu.module @xevm_module{
gpu.func @gemm(%arg0: memref<1024x1024xbf16>, %arg1: memref<1024x1024xbf16>, %arg2: memref<1024x1024xf32>){
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c8 = arith.constant 8 : index
  %c1024 = arith.constant 1024 : index
  %block_id_x = gpu.block_id  x
  %block_id_y = gpu.block_id  y
  %0 = arith.muli %block_id_x, %c8 : index
  %1 = arith.muli %block_id_y, %c16 : index
  %2 = xegpu.create_nd_tdesc %arg2 : memref<1024x1024xf32> ->
    !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  %3 = xegpu.load_nd %2[%0, %1]
    {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
    layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<8x16xf32>
  %5 = xegpu.create_nd_tdesc %arg0: memref<1024x1024xbf16>
      -> !xegpu.tensor_desc<8x16xbf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  %6 = xegpu.create_nd_tdesc %arg1 : memref<1024x1024xbf16>
      -> !xegpu.tensor_desc<16x16xbf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>

  %4 = scf.for %arg3 = %c0 to %c1024 step %c16 iter_args(%arg4 = %3) -> (vector<8x16xf32>) {
    %7 = xegpu.load_nd %5[%0, %arg3]
      {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
       layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
      : !xegpu.tensor_desc<8x16xbf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<8x16xbf16>
    %8 = xegpu.load_nd %6[%arg3, %1]
      {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>,
       layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>}
      : !xegpu.tensor_desc<16x16xbf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>> -> vector<16x16xbf16>

    %9 = xegpu.dpas %7, %8, %arg4
      {layout_a = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
       layout_b = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>,
       layout_cd = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
       layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
      : vector<8x16xbf16>, vector<16x16xbf16>, vector<8x16xf32> -> vector<8x16xf32>

    scf.yield %9 : vector<8x16xf32>
  } {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
  xegpu.store_nd %4, %2[%0, %1] {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}: vector<8x16xf32>,
    !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  gpu.return
}

// CHECK-LABEL: gpu.func @gemm_with_preop
// CHECK-DAG  : %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG  : %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG  : %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG  : %[[C1024:.*]] = arith.constant 1024 : index
// CHECK      : %[[CST:.*]] = arith.constant dense<1.000000e+00> : vector<8x1xbf16>
// CHECK-DAG  : %[[BID_X:.*]] = gpu.block_id  x
// CHECK-DAG  : %[[BID_Y:.*]] = gpu.block_id  y
// CHECK-DAG  : %[[MUL_X:.*]] = arith.muli %[[BID_X]], %[[C8]] : index
// CHECK-DAG  : %[[MUL_Y:.*]] = arith.muli %[[BID_Y]], %[[C16]] : index
// CHECK      : %[[TD_C:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
// CHECK      : %[[LOAD_C:.*]] = xegpu.load_nd %[[TD_C]][%[[MUL_X]], %[[MUL_Y]]] : !xegpu.tensor_desc<8x16xf32> -> vector<8xf32>
// CHECK-DAG  : %[[CAST_C:.*]] = vector.shape_cast %[[LOAD_C]] : vector<8xf32> to vector<8x1xf32>
// CHECK-DAG  : %[[TD_A:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<1024x1024xbf16> -> !xegpu.tensor_desc<8x16xbf16>
// CHECK-DAG  : %[[TD_B:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<1024x1024xbf16> -> !xegpu.tensor_desc<16x16xbf16>
// CHECK      : %[[FOR:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C1024]] step %[[C16]] iter_args(%[[ACC:.*]] = %[[CAST_C]]) -> (vector<8x1xf32>) {
// CHECK-DAG  :   %[[LOAD_A:.*]] = xegpu.load_nd %[[TD_A]][%[[MUL_X]], %[[IV]]] : !xegpu.tensor_desc<8x16xbf16> -> vector<8xbf16>
// CHECK      :   %[[CAST_A:.*]] = vector.shape_cast %[[LOAD_A]] : vector<8xbf16> to vector<8x1xbf16>
// CHECK      :   %[[PREOP:.*]] = arith.addf %[[CAST_A]], %[[CST]] : vector<8x1xbf16>
// CHECK-DAG  :   %[[LOAD_B:.*]] = xegpu.load_nd %[[TD_B]][%[[IV]], %[[MUL_Y]]] <{packed}> : !xegpu.tensor_desc<16x16xbf16> -> vector<16xbf16>
// CHECK-DAG  :   %[[CAST_ACC:.*]] = vector.shape_cast %[[ACC]] : vector<8x1xf32> to vector<8xf32>
// CHECK      :   %[[CAST_PREOP:.*]] = vector.shape_cast %[[PREOP]] : vector<8x1xbf16> to vector<8xbf16>
// CHECK      :   %[[DPAS:.*]] = xegpu.dpas %[[CAST_PREOP]], %[[LOAD_B]], %[[CAST_ACC]] : vector<8xbf16>, vector<16xbf16>, vector<8xf32> -> vector<8xf32>
// CHECK      :   %[[CAST_DPAS:.*]] = vector.shape_cast %[[DPAS]] : vector<8xf32> to vector<8x1xf32>
// CHECK      :   scf.yield %[[CAST_DPAS]] : vector<8x1xf32>
// CHECK      : } {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
// CHECK      : %[[CAST_FOR:.*]] = vector.shape_cast %[[FOR]] : vector<8x1xf32> to vector<8xf32>
// CHECK      : xegpu.store_nd %[[CAST_FOR]], %[[TD_C]][%[[MUL_X]], %[[MUL_Y]]] : vector<8xf32>, !xegpu.tensor_desc<8x16xf32>
// CHECK      : gpu.return
gpu.func @gemm_with_preop(%arg0: memref<1024x1024xbf16>, %arg1: memref<1024x1024xbf16>, %arg2: memref<1024x1024xf32>){
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c8 = arith.constant 8 : index
  %c1024 = arith.constant 1024 : index
  %cst = arith.constant  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} dense<1.0> : vector<8x16xbf16>
  %block_id_x = gpu.block_id  x
  %block_id_y = gpu.block_id  y
  %0 = arith.muli %block_id_x, %c8 : index
  %1 = arith.muli %block_id_y, %c16 : index
  %2 = xegpu.create_nd_tdesc %arg2 : memref<1024x1024xf32> ->
    !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  %3 = xegpu.load_nd %2[%0, %1]
    {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
    layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<8x16xf32>
  %5 = xegpu.create_nd_tdesc %arg0: memref<1024x1024xbf16>
      -> !xegpu.tensor_desc<8x16xbf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  %6 = xegpu.create_nd_tdesc %arg1 : memref<1024x1024xbf16>
      -> !xegpu.tensor_desc<16x16xbf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>

  %4 = scf.for %arg3 = %c0 to %c1024 step %c16 iter_args(%arg4 = %3) -> (vector<8x16xf32>) {
    %7 = xegpu.load_nd %5[%0, %arg3]
      {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
       layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
      : !xegpu.tensor_desc<8x16xbf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<8x16xbf16>
    %preop = arith.addf %7, %cst {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<8x16xbf16>
    %8 = xegpu.load_nd %6[%arg3, %1]
      {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>,
       layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>}
      : !xegpu.tensor_desc<16x16xbf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>> -> vector<16x16xbf16>

    %9 = xegpu.dpas %preop, %8, %arg4
      {layout_a = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
       layout_b = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>,
       layout_cd = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
       layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
      : vector<8x16xbf16>, vector<16x16xbf16>, vector<8x16xf32> -> vector<8x16xf32>

    scf.yield %9 : vector<8x16xf32>
  } {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
  xegpu.store_nd %4, %2[%0, %1] {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}: vector<8x16xf32>,
    !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  gpu.return
}

// CHECK-LABEL: gpu.func @gemm_with_postop
// CHECK-DAG  : %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG  : %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG  : %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG  : %[[C1024:.*]] = arith.constant 1024 : index
// CHECK-DAG  : %[[BID_X:.*]] = gpu.block_id  x
// CHECK-DAG  : %[[BID_Y:.*]] = gpu.block_id  y
// CHECK-DAG  : %[[MUL_X:.*]] = arith.muli %[[BID_X]], %[[C8]] : index
// CHECK-DAG  : %[[MUL_Y:.*]] = arith.muli %[[BID_Y]], %[[C16]] : index
// CHECK      : %[[TD_C:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
// CHECK      : %[[LOAD_C:.*]] = xegpu.load_nd %[[TD_C]][%[[MUL_X]], %[[MUL_Y]]] : !xegpu.tensor_desc<8x16xf32> -> vector<8xf32>
// CHECK-DAG  : %[[CAST_C:.*]] = vector.shape_cast %[[LOAD_C]] : vector<8xf32> to vector<8x1xf32>
// CHECK-DAG  : %[[TD_A:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<1024x1024xbf16> -> !xegpu.tensor_desc<8x16xbf16>
// CHECK-DAG  : %[[TD_B:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<1024x1024xbf16> -> !xegpu.tensor_desc<16x16xbf16>
// CHECK      : %[[FOR:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C1024]] step %[[C16]] iter_args(%[[ACC:.*]] = %[[CAST_C]]) -> (vector<8x1xf32>) {
// CHECK-DAG  :   %[[LOAD_A:.*]] = xegpu.load_nd %[[TD_A]][%[[MUL_X]], %[[IV]]] : !xegpu.tensor_desc<8x16xbf16> -> vector<8xbf16>
// CHECK-DAG  :   %[[LOAD_B:.*]] = xegpu.load_nd %[[TD_B]][%[[IV]], %[[MUL_Y]]] <{packed}> : !xegpu.tensor_desc<16x16xbf16> -> vector<16xbf16>
// CHECK-DAG  :   %[[CAST_ACC:.*]] = vector.shape_cast %[[ACC]] : vector<8x1xf32> to vector<8xf32>
// CHECK      :   %[[DPAS:.*]] = xegpu.dpas %[[LOAD_A]], %[[LOAD_B]], %[[CAST_ACC]] : vector<8xbf16>, vector<16xbf16>, vector<8xf32> -> vector<8xf32>
// CHECK      :   %[[CAST_DPAS:.*]] = vector.shape_cast %[[DPAS]] : vector<8xf32> to vector<8x1xf32>
// CHECK      :   scf.yield %[[CAST_DPAS]] : vector<8x1xf32>
// CHECK      : } {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
// CHECK      : %[[POSTOP:.*]] = math.exp %[[FOR]] : vector<8x1xf32>
// CHECK      : %[[CAST_POSTOP:.*]] = vector.shape_cast %[[POSTOP]] : vector<8x1xf32> to vector<8xf32>
// CHECK      : xegpu.store_nd %[[CAST_POSTOP]], %[[TD_C]][%[[MUL_X]], %[[MUL_Y]]] : vector<8xf32>, !xegpu.tensor_desc<8x16xf32>
gpu.func @gemm_with_postop(%arg0: memref<1024x1024xbf16>, %arg1: memref<1024x1024xbf16>, %arg2: memref<1024x1024xf32>){
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c8 = arith.constant 8 : index
  %c1024 = arith.constant 1024 : index
  %block_id_x = gpu.block_id  x
  %block_id_y = gpu.block_id  y
  %0 = arith.muli %block_id_x, %c8 : index
  %1 = arith.muli %block_id_y, %c16 : index
  %2 = xegpu.create_nd_tdesc %arg2 : memref<1024x1024xf32> ->
    !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  %3 = xegpu.load_nd %2[%0, %1]
    {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
    layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<8x16xf32>
  %5 = xegpu.create_nd_tdesc %arg0: memref<1024x1024xbf16>
      -> !xegpu.tensor_desc<8x16xbf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  %6 = xegpu.create_nd_tdesc %arg1 : memref<1024x1024xbf16>
      -> !xegpu.tensor_desc<16x16xbf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>

  %4 = scf.for %arg3 = %c0 to %c1024 step %c16 iter_args(%arg4 = %3) -> (vector<8x16xf32>) {
    %7 = xegpu.load_nd %5[%0, %arg3]
      {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
       layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
      : !xegpu.tensor_desc<8x16xbf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<8x16xbf16>
    %8 = xegpu.load_nd %6[%arg3, %1]
      {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>,
       layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>}
      : !xegpu.tensor_desc<16x16xbf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>> -> vector<16x16xbf16>

    %9 = xegpu.dpas %7, %8, %arg4
      {layout_a = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
       layout_b = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>,
       layout_cd = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
       layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
      : vector<8x16xbf16>, vector<16x16xbf16>, vector<8x16xf32> -> vector<8x16xf32>

    scf.yield %9 : vector<8x16xf32>
  } {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
  %postop = math.exp %4 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<8x16xf32>
  xegpu.store_nd %postop, %2[%0, %1] {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}: vector<8x16xf32>,
    !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  gpu.return
}

}
