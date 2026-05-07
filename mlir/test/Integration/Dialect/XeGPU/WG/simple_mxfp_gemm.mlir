// RUN: mlir-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup zebin-chip=cri" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// XFAIL: *
#a = #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 512], inst_data = [8, 64], lane_layout = [1, 16], lane_data = [1, 1]>
#b_packed = #xegpu.layout<sg_layout = [8, 8], sg_data = [256, 16], inst_data = [32, 16], lane_layout = [1, 16], lane_data = [4, 1]>
#b = #xegpu.layout<sg_layout = [8, 8], sg_data = [512, 16], inst_data = [64, 16], lane_layout = [1, 16], lane_data = [8, 1]>
#c = #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16], inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>
#a_scale = #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16], inst_data = [8, 2], lane_layout = [8, 1], lane_data = [1, 1]>
#b_scale = #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16], inst_data = [2, 16], lane_layout = [1, 16], lane_data = [1, 1]>

gpu.module @test {
  gpu.func @gemm_mxfp(%arg0: memref<1024x4096xf4E2M1FN>, %arg1: memref<2048x1024xui8>, %arg2: memref<1024x128xf8E8M0FNU>, %arg3: memref<128x1024xf8E8M0FNU>, %arg4: memref<1024x1024xf32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y
    %0 = arith.muli %block_id_x, %c128 : index
    %1 = arith.muli %block_id_y, %c128 : index

    %a_tdesc = xegpu.create_nd_tdesc %arg0 : memref<1024x4096xf4E2M1FN> -> !xegpu.tensor_desc<128x512xf4E2M1FN>
    %bp_tdesc = xegpu.create_nd_tdesc %arg1 : memref<2048x1024xui8> -> !xegpu.tensor_desc<256x128xui8>
    // load_nd with offset
    %a = xegpu.load_nd %a_tdesc[%0, %c0] {layout = #a}: !xegpu.tensor_desc<128x512xf4E2M1FN> -> vector<128x512xf4E2M1FN>
    %bp = xegpu.load_nd %bp_tdesc[%c0, %1] {layout = #b_packed}: !xegpu.tensor_desc<256x128xui8> -> vector<256x128xui8>

    // Bitcast to fp4: 256x128 uint8 -> 256x256 fp4 (each uint8 holds 2 fp4 values)
    %b_bitcast = vector.bitcast %bp : vector<256x128xui8> to vector<256x256xf4E2M1FN>

    // De-interleave: extract even and odd columns
    // Even columns (indices 0, 2, 4, ..., 254) -> first half
    // Odd columns (indices 1, 3, 5, ..., 255) -> second half
    %b_even, %b_odd = vector.deinterleave %b_bitcast : vector<256x256xf4E2M1FN> -> vector<256x128xf4E2M1FN>

    // Reconstruct 512x128 by interleaving even/odd rows:
    // Transpose to move the row dim to trailing position, interleave, transpose back.
    %b_even_t = vector.transpose %b_even, [1, 0] : vector<256x128xf4E2M1FN> to vector<128x256xf4E2M1FN>
    %b_odd_t = vector.transpose %b_odd, [1, 0] : vector<256x128xf4E2M1FN> to vector<128x256xf4E2M1FN>
    %b_interleaved = vector.interleave %b_even_t, %b_odd_t : vector<128x256xf4E2M1FN> -> vector<128x512xf4E2M1FN>
    %b = vector.transpose %b_interleaved, [1, 0] : vector<128x512xf4E2M1FN> to vector<512x128xf4E2M1FN>

    %cd_tdesc = xegpu.create_nd_tdesc %arg4 : memref<1024x1024xf32> -> !xegpu.tensor_desc<128x128xf32, #c>
    %c = xegpu.load_nd %cd_tdesc[%0, %1] {layout = #c}: !xegpu.tensor_desc<128x128xf32, #c> -> vector<128x128xf32>

    %a_scale_tdesc = xegpu.create_nd_tdesc %arg2 : memref<1024x128xf8E8M0FNU> -> !xegpu.tensor_desc<128x16xf8E8M0FNU>
    %scale_a = xegpu.load_nd %a_scale_tdesc[%0, %c0] {layout = #a_scale}: !xegpu.tensor_desc<128x16xf8E8M0FNU> -> vector<128x16xf8E8M0FNU>

    %b_scale_tdesc = xegpu.create_nd_tdesc %arg3 : memref<128x1024xf8E8M0FNU> -> !xegpu.tensor_desc<16x128xf8E8M0FNU>
    %scale_b = xegpu.load_nd %b_scale_tdesc[%c0, %1] {layout = #b_scale}: !xegpu.tensor_desc<16x128xf8E8M0FNU> -> vector<16x128xf8E8M0FNU>

    %d = xegpu.dpas_mx %a, %b, %c scale_a = %scale_a scale_b = %scale_b
          {layout_a = #a,
           layout_b = #b,
           layout_cd = #c,
           layout_a_scale = #a_scale,
           layout_b_scale = #b_scale}
        : vector<128x512xf4E2M1FN>, vector<512x128xf4E2M1FN>,
          vector<128x128xf32>,
          vector<128x16xf8E8M0FNU>, vector<16x128xf8E8M0FNU>
        -> vector<128x128xf32>

    // store_nd with offset
    xegpu.store_nd %d, %cd_tdesc[%0, %1] {layout = #c} : vector<128x128xf32>, !xegpu.tensor_desc<128x128xf32, #c>
    gpu.return
  }
}
