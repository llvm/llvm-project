func.func @gemm_b_transpose(%arg0: memref<256x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<256x256xf32>) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c256 = arith.constant 256 : index
  %0 = xegpu.create_nd_tdesc %arg2 : memref<256x256xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  %1 = xegpu.load_nd %0[%c0, %c0]  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<8x16xf32>
  %2 = xegpu.create_nd_tdesc %arg0 : memref<256x256xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  %3 = xegpu.create_nd_tdesc %arg1 : memref<256x256xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2]>>
  %4 = scf.for %arg3 = %c0 to %c256 step %c16 iter_args(%arg4 = %1) -> (vector<8x16xf32>) {
    %5 = xegpu.load_nd %2[%c0, %arg3]  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : !xegpu.tensor_desc<8x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<8x16xf16>
    %6 = xegpu.load_nd %3[%c0, %arg3]  {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2]>} : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2]>> -> vector<16x16xf16>
    %7 = vector.transpose %6, [1, 0] {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>} : vector<16x16xf16> to vector<16x16xf16>
    %8 = xegpu.dpas %5, %7, %arg4 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    scf.yield %8 : vector<8x16xf32>
  } {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
  xegpu.store_nd %4, %0[%c0, %c0]  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  return
}
