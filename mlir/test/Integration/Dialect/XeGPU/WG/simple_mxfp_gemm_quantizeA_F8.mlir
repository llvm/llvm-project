// RUN: mlir-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup zebin-chip=cri" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// XFAIL: *

// mx-fp8 counterpart of simple_mxfp_gemm_quantizeA_F4.mlir. A is loaded as bf16
// and quantized in-kernel to f8E5M2 plus an f8E8M0 scale per 32 elements along
// K; B and its scale are passed in pre-quantized.
//
// Two differences from the fp4 version. An fp8 element is a whole byte, so B is
// loaded directly and needs no bitcast / deinterleave / interleave
// reconstruction. And the scale divides by the largest power of two E5M2 can
// represent, 2^15, where E2M1 divides by 4.

// Note: layouts used by dpas_mx need to match HW constaint. Otherwise dpas_mx is not unrolled.
#a = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 1024], inst_data = [8, 32], lane_layout = [1, 16], lane_data = [1, 2]>
#a_ld = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 1024], inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>
#b = #xegpu.layout<sg_layout = [2, 2], sg_data = [1024, 16], inst_data = [32, 16], lane_layout = [1, 16], lane_data = [4, 1]>
#c = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 16], inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>
// Note: inst_data is chosen to utilize 2D block load
#b_scale = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 16], inst_data = [32, 16], lane_layout = [1, 16], lane_data = [1, 1]>
// Note: scales for dpas_mx needs separate layouts with inst_data to match HW constraint. Otherwise dpas_mx is not unrolled
#dpas_a_scale = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 32], inst_data = [8, 1], lane_layout = [8, 1], lane_data = [1, 1]>
#dpas_b_scale = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 16], inst_data = [1, 16], lane_layout = [1, 16], lane_data = [1, 1]>


module @gemm attributes {gpu.container_module} {
  gpu.module @kernel {
    // A is loaded as bf16 and quantized in-place to mx-fp8 (f8E5M2 + f8E8M0
    // scale) along the K dimension with block size 32. B and its scale are
    // passed in pre-quantized. The quantized values are then consumed by
    // xegpu.dpas_mx.
    gpu.func @gemm_mxfp(%arg0: memref<256x4096xbf16>, %arg1: memref<4096x256xf8E5M2>, %arg3: memref<128x256xf8E8M0FNU>, %arg4: memref<256x256xf32>) kernel {
      %c0 = arith.constant 0 : index
      %mstep = arith.constant 32 : index
      %nstep = arith.constant 32 : index
      %kstep = arith.constant 1024 : index
      %kbound = arith.constant 4096 : index
      %kscalestep = arith.constant 32 : index
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %m = arith.muli %block_id_x, %mstep : index
      %n = arith.muli %block_id_y, %nstep : index

      %a_tdesc = xegpu.create_nd_tdesc %arg0 : memref<256x4096xbf16> -> !xegpu.tensor_desc<32x1024xbf16>
      %b_tdesc = xegpu.create_nd_tdesc %arg1 : memref<4096x256xf8E5M2> -> !xegpu.tensor_desc<1024x32xf8E5M2>
      %b_scale_tdesc = xegpu.create_nd_tdesc %arg3 : memref<128x256xf8E8M0FNU> -> !xegpu.tensor_desc<32x32xf8E8M0FNU>

      // Load initial C
      %cd_tdesc = xegpu.create_nd_tdesc %arg4 : memref<256x256xf32> -> !xegpu.tensor_desc<32x32xf32, #c>
      %c_init = xegpu.load_nd %cd_tdesc[%m, %n] {layout = #c}: !xegpu.tensor_desc<32x32xf32, #c> -> vector<32x32xf32>

      %res:2 = scf.for %k = %c0 to %kbound step %kstep
        iter_args(%c_partial = %c_init, %kscale = %c0) -> (vector<32x32xf32>, index) {
        // -------- Load A (bf16) --------
        %a_bf16 = xegpu.load_nd %a_tdesc[%m, %k] {layout = #a_ld}: !xegpu.tensor_desc<32x1024xbf16> -> vector<32x1024xbf16>

        // -------- Quantize A: bf16 -> fp8 + f8E8M0 scale (block_size=32 along K) --------
        // 1) abs and reduce-max per block of 32 along K dim using vector ops.
        %a_abs = math.absf %a_bf16 : vector<32x1024xbf16>
        %a_abs_r = vector.shape_cast %a_abs : vector<32x1024xbf16> to vector<32x32x32xbf16>
        %a_neg_inf_i = arith.constant dense<0xFF80> : vector<32x32xi16>
        %a_neg_inf = arith.bitcast %a_neg_inf_i : vector<32x32xi16> to vector<32x32xbf16>
        %a_amax = vector.multi_reduction <maximumf>, %a_abs_r, %a_neg_inf [2]
            : vector<32x32x32xbf16> to vector<32x32xbf16>

        // 2) Largest power-of-two <= amax: mask out mantissa bits of bf16.
        %a_amax_i16 = arith.bitcast %a_amax : vector<32x32xbf16> to vector<32x32xi16>
        %a_exp_mask = arith.constant dense<0x7F80> : vector<32x32xi16>
        %a_pow2_i16 = arith.andi %a_amax_i16, %a_exp_mask : vector<32x32xi16>
        %a_pow2 = arith.bitcast %a_pow2_i16 : vector<32x32xi16> to vector<32x32xbf16>

        // 3) Divide by largest power-of-two representable by E5M2 (= 2^15).
        //    E5M2's largest finite value is 57344 = 1.75 * 2^15, so 2^15 is the
        //    largest power of two it can hold; E2M1 uses 4.0 for the same reason.
        %a_e5m2_max = arith.constant dense<3.276800e+04> : vector<32x32xbf16>
        %a_scale_bf16 = arith.divf %a_pow2, %a_e5m2_max : vector<32x32xbf16>

        // 4) Truncate scale to f8E8M0FNU.
        %a_scale = arith.truncf %a_scale_bf16 : vector<32x32xbf16> to vector<32x32xf8E8M0FNU>

        // 5) Broadcast the per-block scale across the block (32 elements along K).
        //    vector.broadcast can only prepend leading dims, so we broadcast onto a
        //    leading 32 dim, transpose it to the trailing position, then shape_cast.
        %a_scale_lead = vector.broadcast %a_scale
            : vector<32x32xf8E8M0FNU> to vector<32x32x32xf8E8M0FNU>
        %a_scale_t = vector.transpose %a_scale_lead, [1, 2, 0]
            : vector<32x32x32xf8E8M0FNU> to vector<32x32x32xf8E8M0FNU>
        %a_scale_full = vector.shape_cast %a_scale_t
            : vector<32x32x32xf8E8M0FNU> to vector<32x1024xf8E8M0FNU>

        // 6) Scaled truncf to fp8 (to_nearest_even).
        %a = arith.scaling_truncf %a_bf16, %a_scale_full
            : vector<32x1024xbf16>, vector<32x1024xf8E8M0FNU> to vector<32x1024xf8E5M2>

        // -------- Load B (mx-fp8) --------
        // An fp8 element is a whole byte, so B is loaded in its logical shape and
        // needs no unpacking. B is indexed by %k for the same reason.
        %b = xegpu.load_nd %b_tdesc[%k, %n] {layout = #b}: !xegpu.tensor_desc<1024x32xf8E5M2> -> vector<1024x32xf8E5M2>

        %scale_b = xegpu.load_nd %b_scale_tdesc[%kscale, %n] {layout = #b_scale}: !xegpu.tensor_desc<32x32xf8E8M0FNU> -> vector<32x32xf8E8M0FNU>
        %new_c_partial = xegpu.dpas_mx %a, %b, %c_partial scale_a = %a_scale scale_b = %scale_b
              {layout_a = #a,
               layout_b = #b,
               layout_cd = #c,
               layout_a_scale = #dpas_a_scale,
               layout_b_scale = #dpas_b_scale}
            : (vector<32x1024xf8E5M2>, vector<1024x32xf8E5M2>,
               vector<32x32xf32>,
               vector<32x32xf8E8M0FNU>, vector<32x32xf8E8M0FNU>)
            -> vector<32x32xf32>

        // b_scale takes a different step compared to a and b.
        %new_kscale = arith.addi %kscale, %kscalestep : index
        scf.yield %new_c_partial, %new_kscale : vector<32x32xf32>, index
      }

      // store_nd with offset
      xegpu.store_nd %res#0, %cd_tdesc[%m, %n] {layout = #c} : vector<32x32xf32>, !xegpu.tensor_desc<32x32xf32, #c>
      gpu.return
    }
  }

  func.func @test(%a: memref<256x4096xbf16>, %b: memref<4096x256xf8E5M2>, %b_scale: memref<128x256xf8E8M0FNU>, %c: memref<256x256xf32>) -> memref<256x256xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index

    %memref_a = gpu.alloc() : memref<256x4096xbf16>
    gpu.memcpy %memref_a, %a : memref<256x4096xbf16>, memref<256x4096xbf16>

    %memref_b = gpu.alloc() : memref<4096x256xf8E5M2>
    gpu.memcpy %memref_b, %b : memref<4096x256xf8E5M2>, memref<4096x256xf8E5M2>

    %memref_c = gpu.alloc() : memref<256x256xf32>
    gpu.memcpy %memref_c, %c : memref<256x256xf32>, memref<256x256xf32>

    %memref_b_scale = gpu.alloc() : memref<128x256xf8E8M0FNU>
    gpu.memcpy %memref_b_scale, %b_scale : memref<128x256xf8E8M0FNU>, memref<128x256xf8E8M0FNU>

    gpu.launch_func @kernel::@gemm_mxfp blocks in (%c8, %c8, %c1) threads in (%c64, %c1, %c1)
    args(%memref_a : memref<256x4096xbf16>, %memref_b : memref<4096x256xf8E5M2>, %memref_b_scale : memref<128x256xf8E8M0FNU>, %memref_c : memref<256x256xf32>)
    gpu.dealloc %memref_a : memref<256x4096xbf16>
    gpu.dealloc %memref_b : memref<4096x256xf8E5M2>
    gpu.dealloc %memref_b_scale : memref<128x256xf8E8M0FNU>

    %res = memref.alloc() : memref<256x256xf32>
    gpu.memcpy %res, %memref_c : memref<256x256xf32>, memref<256x256xf32>
    gpu.dealloc %memref_c : memref<256x256xf32>
    return %res : memref<256x256xf32>
  }

  func.func @main() attributes {llvm.emit_c_interface} {

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c4K = arith.constant 4096 : index
    %c1bf16 = arith.constant 1.0 : bf16
    %c1f8E5M2 = arith.constant 1.0 : f8E5M2
    %c0f32 = arith.constant 0.0 : f32
    %c1f8E8M0FNU = arith.constant 1.0 : f8E8M0FNU

    %A = memref.alloc() : memref<256x4096xbf16>
    scf.for %i = %c0 to %c256 step %c1 {
      scf.for %j = %c0 to %c4K step %c1 {
        memref.store %c1bf16, %A[%i, %j] : memref<256x4096xbf16>
      }
    }

    %B = memref.alloc() : memref<4096x256xf8E5M2>
    scf.for %i = %c0 to %c4K step %c1 {
      scf.for %j = %c0 to %c256 step %c1 {
        memref.store %c1f8E5M2, %B[%i, %j] : memref<4096x256xf8E5M2>
      }
    }

    %C = memref.alloc() : memref<256x256xf32>
    scf.for %i = %c0 to %c256 step %c1 {
      scf.for %j = %c0 to %c256 step %c1 {
        memref.store %c0f32, %C[%i, %j] : memref<256x256xf32>
      }
    }

    %B_scale = memref.alloc() : memref<128x256xf8E8M0FNU>
    scf.for %i = %c0 to %c128 step %c1 {
      scf.for %j = %c0 to %c256 step %c1 {
        memref.store %c1f8E8M0FNU, %B_scale[%i, %j] : memref<128x256xf8E8M0FNU>
      }
    }

    // A is all 1.0, so each block's amax is 1.0, the scale is 2^-15 and A
    // quantizes to 2^15 exactly; B is 1.0 with a unit scale. Every product is
    // therefore 1.0 and each output element is K.
    %c4Kf = arith.constant 4096.0 : f32
    %C_ref = memref.alloc() : memref<256x256xf32>
    scf.for %i = %c0 to %c256 step %c1 {
      scf.for %j = %c0 to %c256 step %c1 {
        memref.store %c4Kf, %C_ref[%i, %j] : memref<256x256xf32>
      }
    }

    %C_res = call @test(%A, %B, %B_scale, %C) : (memref<256x4096xbf16>, memref<4096x256xf8E5M2>, memref<128x256xf8E8M0FNU>, memref<256x256xf32>) -> memref<256x256xf32>
    %C_cast = memref.cast %C_res : memref<256x256xf32> to memref<*xf32>
    %C_ref_cast = memref.cast %C_ref : memref<256x256xf32> to memref<*xf32>
    %diff = call @verifyMemRefF32(%C_cast, %C_ref_cast) : (memref<*xf32>, memref<*xf32>) -> i64
    call @printI64(%diff) : (i64) -> ()
    //call @printMemrefF32(%C_cast) : (memref<*xf32>) -> ()

    // CHECK: 0
    memref.dealloc %A : memref<256x4096xbf16>
    memref.dealloc %B : memref<4096x256xf8E5M2>
    memref.dealloc %B_scale : memref<128x256xf8E8M0FNU>
    memref.dealloc %C : memref<256x256xf32>
    memref.dealloc %C_res : memref<256x256xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printI64(i64)
  func.func private @verifyMemRefF32(memref<*xf32>, memref<*xf32>) -> i64 attributes {llvm.emit_c_interface}
}
