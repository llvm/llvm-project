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
}
