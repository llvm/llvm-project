// RUN: mlir-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup zebin-chip=cri" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// XFAIL: *
#a = #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 512], inst_data = [8, 64], lane_layout = [1, 16], lane_data = [1, 4]>
#b_packed = #xegpu.layout<sg_layout = [8, 8], sg_data = [256, 16], inst_data = [32, 16], lane_layout = [1, 16], lane_data = [4, 1]>
#b = #xegpu.layout<sg_layout = [8, 8], sg_data = [512, 16], inst_data = [64, 16], lane_layout = [1, 16], lane_data = [8, 1]>
#c = #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16], inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>
#b_scale_ld = #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16], inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>
#a_scale_dpas = #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16], inst_data = [8, 2], lane_layout = [8, 1], lane_data = [1, 2]>
#b_scale_dpas = #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16], inst_data = [2, 16], lane_layout = [1, 16], lane_data = [1, 1]>

gpu.module @test {
  // A is loaded as bf16 and quantized in-place to mx-fp4 (fp4 + f8E8M0 scale)
  // along the K dimension with block size 32. B and its scale are passed in
  // pre-quantized (packed ui8 fp4 and f8E8M0). The quantized values are then
  // consumed by xegpu.dpas_mx.
  gpu.func @gemm_mxfp(%arg0: memref<1024x4096xbf16>, %arg1: memref<2048x1024xui8>, %arg2: memref<128x1024xf8E8M0FNU>, %arg3: memref<1024x1024xf32>) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y
    %0 = arith.muli %block_id_x, %c128 : index
    %1 = arith.muli %block_id_y, %c128 : index

    // -------- Load A (bf16) --------
    %a_tdesc = xegpu.create_nd_tdesc %arg0 : memref<1024x4096xbf16> -> !xegpu.tensor_desc<128x512xbf16>
    %a_bf16 = xegpu.load_nd %a_tdesc[%0, %c0] {layout = #a}: !xegpu.tensor_desc<128x512xbf16> -> vector<128x512xbf16>

    // -------- Quantize A: bf16 -> fp4 + f8E8M0 scale (block_size=32 along K) --------
    // 1) abs and reduce-max per block of 32 along K dim using vector ops.
    %a_abs = math.absf %a_bf16 : vector<128x512xbf16>
    %a_abs_r = vector.shape_cast %a_abs : vector<128x512xbf16> to vector<128x16x32xbf16>
    %a_neg_inf_i = arith.constant dense<0xFF80> : vector<128x16xi16>
    %a_neg_inf = arith.bitcast %a_neg_inf_i : vector<128x16xi16> to vector<128x16xbf16>
    %a_amax = vector.multi_reduction <maximumf>, %a_abs_r, %a_neg_inf [2]
        : vector<128x16x32xbf16> to vector<128x16xbf16>

    // 2) Largest power-of-two <= amax: mask out mantissa bits of bf16.
    %a_amax_i16 = arith.bitcast %a_amax : vector<128x16xbf16> to vector<128x16xi16>
    %a_exp_mask = arith.constant dense<0x7F80> : vector<128x16xi16>
    %a_pow2_i16 = arith.andi %a_amax_i16, %a_exp_mask : vector<128x16xi16>
    %a_pow2 = arith.bitcast %a_pow2_i16 : vector<128x16xi16> to vector<128x16xbf16>

    // 3) Divide by largest power-of-two representable by E2M1 (= 4.0).
    %a_e2m1_max = arith.constant dense<4.000000e+00> : vector<128x16xbf16>
    %a_scale_bf16 = arith.divf %a_pow2, %a_e2m1_max : vector<128x16xbf16>

    // 4) Truncate scale to f8E8M0FNU.
    %a_scale = arith.truncf %a_scale_bf16 : vector<128x16xbf16> to vector<128x16xf8E8M0FNU>

    // 5) Broadcast the per-block scale across the block (32 elements along K).
    //    vector.broadcast can only prepend leading dims, so we broadcast onto a
    //    leading 32 dim, transpose it to the trailing position, then shape_cast.
    %a_scale_lead = vector.broadcast %a_scale
        : vector<128x16xf8E8M0FNU> to vector<32x128x16xf8E8M0FNU>
    %a_scale_t = vector.transpose %a_scale_lead, [1, 2, 0]
        : vector<32x128x16xf8E8M0FNU> to vector<128x16x32xf8E8M0FNU>
    %a_scale_full = vector.shape_cast %a_scale_t
        : vector<128x16x32xf8E8M0FNU> to vector<128x512xf8E8M0FNU>

    // 6) Scaled truncf to fp4 (to_nearest_even).
    %a = arith.scaling_truncf %a_bf16, %a_scale_full
        : vector<128x512xbf16>, vector<128x512xf8E8M0FNU> to vector<128x512xf4E2M1FN>

    // -------- Load B (packed ui8 fp4) and unpack to f4E2M1FN --------
    %bp_tdesc = xegpu.create_nd_tdesc %arg1 : memref<2048x1024xui8> -> !xegpu.tensor_desc<256x128xui8>
    %bp = xegpu.load_nd %bp_tdesc[%c0, %1] {layout = #b_packed}: !xegpu.tensor_desc<256x128xui8> -> vector<256x128xui8>

    // Bitcast to fp4: 256x128 uint8 -> 256x256 fp4 (each uint8 holds 2 fp4 values)
    %b_bitcast = vector.bitcast %bp : vector<256x128xui8> to vector<256x256xf4E2M1FN>

    // De-interleave: extract even and odd columns
    %b_even, %b_odd = vector.deinterleave %b_bitcast : vector<256x256xf4E2M1FN> -> vector<256x128xf4E2M1FN>

    // Reconstruct 512x128 by interleaving even/odd rows.
    %b_even_t = vector.transpose %b_even, [1, 0] : vector<256x128xf4E2M1FN> to vector<128x256xf4E2M1FN>
    %b_odd_t = vector.transpose %b_odd, [1, 0] : vector<256x128xf4E2M1FN> to vector<128x256xf4E2M1FN>
    %b_interleaved = vector.interleave %b_even_t, %b_odd_t : vector<128x256xf4E2M1FN> -> vector<128x512xf4E2M1FN>
    %b = vector.transpose %b_interleaved, [1, 0] : vector<128x512xf4E2M1FN> to vector<512x128xf4E2M1FN>

    // -------- Load B scale --------
    %b_scale_tdesc = xegpu.create_nd_tdesc %arg2 : memref<128x1024xf8E8M0FNU> -> !xegpu.tensor_desc<16x128xf8E8M0FNU>
    %b_scale = xegpu.load_nd %b_scale_tdesc[%c0, %1] {layout = #b_scale_ld}: !xegpu.tensor_desc<16x128xf8E8M0FNU> -> vector<16x128xf8E8M0FNU>

    // -------- Load C and run dpas_mx --------
    %cd_tdesc = xegpu.create_nd_tdesc %arg3 : memref<1024x1024xf32> -> !xegpu.tensor_desc<128x128xf32, #c>
    %c = xegpu.load_nd %cd_tdesc[%0, %1] {layout = #c}: !xegpu.tensor_desc<128x128xf32, #c> -> vector<128x128xf32>

    %d = xegpu.dpas_mx %a, %b, %c scale_a = %a_scale scale_b = %b_scale
          {layout_a = #a,
           layout_b = #b,
           layout_cd = #c,
           layout_a_scale = #a_scale_dpas,
           layout_b_scale = #b_scale_dpas}
        : (vector<128x512xf4E2M1FN>, vector<512x128xf4E2M1FN>,
          vector<128x128xf32>,
          vector<128x16xf8E8M0FNU>, vector<16x128xf8E8M0FNU>)
        -> vector<128x128xf32>

    // store_nd with offset
    xegpu.store_nd %d, %cd_tdesc[%0, %1] {layout = #c} : vector<128x128xf32>, !xegpu.tensor_desc<128x128xf32, #c>
    gpu.return
  }
}
