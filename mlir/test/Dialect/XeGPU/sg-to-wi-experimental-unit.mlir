
// RUN: mlir-opt  --xevm-attach-target='module=xevm_* chip=pvc' \
// --test-xegpu-sg-to-wi-distribute-experimental --split-input-file %s | FileCheck %s

gpu.module @xevm_module {
gpu.func @create_nd_tdesc(%arg0: memref<256x256xf16>) {
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0 : memref<256x256xf16>
    -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  gpu.return
}

gpu.func @cerate_nd_tedesc_nonmemref_source(%arg0: ui64) {
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0, shape : [256, 256], strides : [256, 1] : ui64
    -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  gpu.return
}

gpu.func @load_nd() {
  %c0 = arith.constant 0 : index
  %0 = "some_op"() : () -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  %1 = xegpu.load_nd %0[%c0, %c0] {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<16x16xf16>
  gpu.return
}

gpu.func @store_nd() {
  %c0 = arith.constant 0 : index
  %0 = "some_op"() : () -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  %1 = "some_op"() : () -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  %2 = xegpu.load_nd %0[%c0, %c0] {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<16x16xf16>
  xegpu.store_nd %2, %1[%c0, %c0] {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  gpu.return
}

gpu.func @dpas_op() {
  %c0 = arith.constant 0 : index
  %0 = "some_op"() : () -> !xegpu.tensor_desc<8x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  %1 = "some_op"() : () -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>
  %5 = arith.constant  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    dense<0.0> : vector<8x16xf32>
  %2 = xegpu.load_nd %0[%c0, %c0] {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : !xegpu.tensor_desc<8x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<8x16xf16>
  %3 = xegpu.load_nd %1[%c0, %c0] {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>}
    : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>> -> vector<16x16xf16>
  %4 = xegpu.dpas %2, %3, %5
    {layout_a = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
     layout_b = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
     layout_cd = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32>  -> vector<8x16xf32>
  gpu.return
}

gpu.func @elementwise_op() {
  %c0 = arith.constant 0 : index
  %0 = arith.constant  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    dense<1.0> : vector<16x16xf32>
  %1 = "some_op"() : () -> !xegpu.tensor_desc<16x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  %2 = xegpu.load_nd %1[%c0, %c0] {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : !xegpu.tensor_desc<16x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<16x16xf32>
  %3 = arith.addf %0, %2
    {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : vector<16x16xf32>
  gpu.return
}
}
