// RUN: mlir-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup zebin-chip=cri"
// RUN-DISABLED: mlir-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup zebin-chip=cri" \
// RUN-DISABLED: | mlir-runner \
// RUN-DISABLED:   --shared-libs=%mlir_levelzero_runtime \
// RUN-DISABLED:   --shared-libs=%mlir_runner_utils \
// RUN-DISABLED:   --shared-libs=%mlir_c_runner_utils \
// RUN-DISABLED:   --entry-point-result=void \
// RUN-DISABLED: | FileCheck --check-prefix=MISMATCH %s

// Note: layouts used by dpas_mx need to match HW constaint. Otherwise dpas_mx is not unrolled.
#a = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 1024], inst_data = [8, 64], lane_layout = [1, 16], lane_data = [1, 4]>
#a_ld = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 1024], inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>
#b_packed = #xegpu.layout<sg_layout = [2, 2], sg_data = [512, 16], inst_data = [32, 16], lane_layout = [1, 16], lane_data = [4, 1]>
#b = #xegpu.layout<sg_layout = [2, 2], sg_data = [1024, 16], inst_data = [64, 16], lane_layout = [1, 16], lane_data = [8, 1]>
#c = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 16], inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>
// Note: inst_data is chosen to utilize 2D block load
#b_scale = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 16], inst_data = [32, 16], lane_layout = [1, 16], lane_data = [1, 1]>
// Note: scales for dpas_mx needs separate layouts with inst_data to match HW constraint. Otherwise dpas_mx is not unrolled
#dpas_a_scale = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 32], inst_data = [8, 2], lane_layout = [8, 1], lane_data = [1, 1]>
#dpas_b_scale = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 16], inst_data = [2, 16], lane_layout = [1, 16], lane_data = [1, 1]>


module @gemm attributes {gpu.container_module} {
  gpu.module @kernel {
    // A is loaded as bf16 and quantized in-place to mx-fp4 (fp4 + f8E8M0 scale)
    // along the K dimension with block size 32. B and its scale are passed in
    // pre-quantized (packed ui8 fp4 and f8E8M0). The quantized values are then
    // consumed by xegpu.dpas_mx.
    gpu.func @gemm_mxfp(%arg0: memref<256x4096xbf16>, %arg1: memref<2048x256xi8>, %arg3: memref<128x256xf8E8M0FNU>, %arg4: memref<256x256xf32>) kernel {
      %c0 = arith.constant 0 : index
      %mstep = arith.constant 32 : index
      %nstep = arith.constant 32 : index
      %kstep = arith.constant 1024 : index
      %kbound = arith.constant 4096 : index
      %kbstep = arith.constant 512 : index
      %kscalestep = arith.constant 32 : index
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %m = arith.muli %block_id_x, %mstep : index
      %n = arith.muli %block_id_y, %nstep : index

      %a_tdesc = xegpu.create_nd_tdesc %arg0 : memref<256x4096xbf16> -> !xegpu.tensor_desc<32x1024xbf16>
      %bp_tdesc = xegpu.create_nd_tdesc %arg1 : memref<2048x256xi8> -> !xegpu.tensor_desc<512x32xi8>
      %b_scale_tdesc = xegpu.create_nd_tdesc %arg3 : memref<128x256xf8E8M0FNU> -> !xegpu.tensor_desc<32x32xf8E8M0FNU>

      // Load initial C
      %cd_tdesc = xegpu.create_nd_tdesc %arg4 : memref<256x256xf32> -> !xegpu.tensor_desc<32x32xf32, #c>
      %c_init = xegpu.load_nd %cd_tdesc[%m, %n] <{layout = #c}>: !xegpu.tensor_desc<32x32xf32, #c> -> vector<32x32xf32>

      %res:3 = scf.for %k = %c0 to %kbound step %kstep
        iter_args(%c_partial = %c_init, %kb = %c0, %kscale = %c0) -> (vector<32x32xf32>, index, index) {
        // -------- Load A (bf16) --------
        %a_bf16 = xegpu.load_nd %a_tdesc[%m, %k] <{layout = #a_ld}>: !xegpu.tensor_desc<32x1024xbf16> -> vector<32x1024xbf16>

        // -------- Quantize A: bf16 -> fp4 + f8E8M0 scale (block_size=32 along K) --------
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

        // 3) Divide by largest power-of-two representable by E2M1 (= 4.0).
        %a_e2m1_max = arith.constant dense<4.000000e+00> : vector<32x32xbf16>
        %a_scale_bf16 = arith.divf %a_pow2, %a_e2m1_max : vector<32x32xbf16>

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

        // 6) Scaled truncf to fp4 (to_nearest_even).
        %a = arith.scaling_truncf %a_bf16, %a_scale_full
            : vector<32x1024xbf16>, vector<32x1024xf8E8M0FNU> to vector<32x1024xf4E2M1FN>

        %bp = xegpu.load_nd %bp_tdesc[%kb, %n] <{layout = #b_packed}>: !xegpu.tensor_desc<512x32xi8> -> vector<512x32xi8>

        // Bitcast to fp4: 512x32 uint8 -> 512x64 fp4 (each uint8 holds 2 fp4 values)
        %b_bitcast = vector.bitcast %bp : vector<512x32xi8> to vector<512x64xf4E2M1FN>

        // De-interleave: extract even and odd columns
        // Even columns (indices 0, 2, 4, ..., 62) -> first half
        // Odd columns (indices 1, 3, 5, ..., 63) -> second half
        %b_even, %b_odd = vector.deinterleave %b_bitcast : vector<512x64xf4E2M1FN> -> vector<512x32xf4E2M1FN>

        // Reconstruct 1024x32 by interleaving even/odd rows:
        // Transpose to move the row dim to trailing position, interleave, transpose back.
        %b_even_t = vector.transpose %b_even, [1, 0] : vector<512x32xf4E2M1FN> to vector<32x512xf4E2M1FN>
        %b_odd_t = vector.transpose %b_odd, [1, 0] : vector<512x32xf4E2M1FN> to vector<32x512xf4E2M1FN>
        %b_interleaved = vector.interleave %b_even_t, %b_odd_t : vector<32x512xf4E2M1FN> -> vector<32x1024xf4E2M1FN>
        %b = vector.transpose %b_interleaved, [1, 0] : vector<32x1024xf4E2M1FN> to vector<1024x32xf4E2M1FN>


        %scale_b = xegpu.load_nd %b_scale_tdesc[%kscale, %n] <{layout = #b_scale}>: !xegpu.tensor_desc<32x32xf8E8M0FNU> -> vector<32x32xf8E8M0FNU>
        %new_c_partial = xegpu.dpas_mx %a, %b, %c_partial scale_a = %a_scale scale_b = %scale_b
              <{layout_a = #a,
               layout_b = #b,
               layout_cd = #c,
               layout_a_scale = #dpas_a_scale,
               layout_b_scale = #dpas_b_scale}>
            : (vector<32x1024xf4E2M1FN>, vector<1024x32xf4E2M1FN>,
               vector<32x32xf32>,
               vector<32x32xf8E8M0FNU>, vector<32x32xf8E8M0FNU>)
            -> vector<32x32xf32>

        // b, a_scale and b_scale take different steps compared to a
        // compute adjusted k index for those tiles.
        %new_kb = arith.addi %kb, %kbstep : index
        %new_kscale = arith.addi %kscale, %kscalestep : index
        scf.yield %new_c_partial, %new_kb, %new_kscale : vector<32x32xf32>, index, index
      }

      // store_nd with offset
      xegpu.store_nd %res#0, %cd_tdesc[%m, %n] <{layout = #c}> : vector<32x32xf32>, !xegpu.tensor_desc<32x32xf32, #c>
      gpu.return
    }
  }

  func.func @test(%a: memref<256x4096xbf16>, %b: memref<2048x256xi8>, %b_scale: memref<128x256xf8E8M0FNU>, %c: memref<256x256xf32>) -> memref<256x256xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index

    %memref_a = gpu.alloc() : memref<256x4096xbf16>
    gpu.memcpy %memref_a, %a : memref<256x4096xbf16>, memref<256x4096xbf16>

    %memref_b = gpu.alloc() : memref<2048x256xi8>
    gpu.memcpy %memref_b, %b : memref<2048x256xi8>, memref<2048x256xi8>

    %memref_c = gpu.alloc() : memref<256x256xf32>
    gpu.memcpy %memref_c, %c : memref<256x256xf32>, memref<256x256xf32>

    %memref_b_scale = gpu.alloc() : memref<128x256xf8E8M0FNU>
    gpu.memcpy %memref_b_scale, %b_scale : memref<128x256xf8E8M0FNU>, memref<128x256xf8E8M0FNU>

    gpu.launch_func @kernel::@gemm_mxfp blocks in (%c8, %c8, %c1) threads in (%c64, %c1, %c1)
    args(%memref_a : memref<256x4096xbf16>, %memref_b : memref<2048x256xi8>, %memref_b_scale : memref<128x256xf8E8M0FNU>, %memref_c : memref<256x256xf32>)
    gpu.dealloc %memref_a : memref<256x4096xbf16>
    gpu.dealloc %memref_b : memref<2048x256xi8>
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
    %c2K = arith.constant 2048 : index
    %c4K = arith.constant 4096 : index
    %c0f32 = arith.constant 0.0 : f32

    // Three block scales for B, one per K block of 32. Per the MX spec a scale
    // is a power of two, so folding it into the reference cannot round.
    %sc = memref.alloc() : memref<3xf8E8M0FNU>
    %scf32 = memref.alloc() : memref<3xf32>
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %s0 = arith.constant 0.5 : f8E8M0FNU
    %s1 = arith.constant 1.0 : f8E8M0FNU
    %s2 = arith.constant 2.0 : f8E8M0FNU
    memref.store %s0, %sc[%c0] : memref<3xf8E8M0FNU>
    memref.store %s1, %sc[%i1] : memref<3xf8E8M0FNU>
    memref.store %s2, %sc[%i2] : memref<3xf8E8M0FNU>
    %sf0 = arith.constant 0.5 : f32
    %sf1 = arith.constant 1.0 : f32
    %sf2 = arith.constant 2.0 : f32
    memref.store %sf0, %scf32[%c0] : memref<3xf32>
    memref.store %sf1, %scf32[%i1] : memref<3xf32>
    memref.store %sf2, %scf32[%i2] : memref<3xf32>

    %c8 = arith.constant 8 : index
    %c3 = arith.constant 3 : index
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index

    %A_f32 = memref.alloc() : memref<256x4096xf32>
    %B_f32 = memref.alloc() : memref<4096x256xf32>

    // A is bf16 on input and quantized in the kernel. Each block of 32 cycles
    // through one family of four values, so its amax is the family maximum and
    // the derived scale is exactly 1, 1/2, 1/4 or 1/8 depending on the block.
    // Rescaling maps every family onto {0, 2, 4, 6}, so the quantization is
    // lossless and never reaches the clamping path.
    // The 8 magnitudes e2m1 can represent, indexed by their e2m1 bit pattern.
    %lut = memref.alloc() : memref<8xf32>
    %i3 = arith.constant 3 : index
    %i4 = arith.constant 4 : index
    %i5 = arith.constant 5 : index
    %i6 = arith.constant 6 : index
    %i7 = arith.constant 7 : index
    %f0 = arith.constant 0.0 : f32
    %f1 = arith.constant 0.5 : f32
    %f2 = arith.constant 1.0 : f32
    %f3 = arith.constant 1.5 : f32
    %f4 = arith.constant 2.0 : f32
    %f5 = arith.constant 3.0 : f32
    %f6 = arith.constant 4.0 : f32
    %f7 = arith.constant 6.0 : f32
    memref.store %f0, %lut[%c0] : memref<8xf32>
    memref.store %f1, %lut[%i1] : memref<8xf32>
    memref.store %f2, %lut[%i2] : memref<8xf32>
    memref.store %f3, %lut[%i3] : memref<8xf32>
    memref.store %f4, %lut[%i4] : memref<8xf32>
    memref.store %f5, %lut[%i5] : memref<8xf32>
    memref.store %f6, %lut[%i6] : memref<8xf32>
    memref.store %f7, %lut[%i7] : memref<8xf32>

    // A's per-K-block divisor, matching what the MX rule derives from the
    // block's amax. Powers of two, so dividing cannot round.
    %adiv = memref.alloc() : memref<3xf32>
    %ad0 = arith.constant 1.0 : f32
    %ad1 = arith.constant 2.0 : f32
    %ad2 = arith.constant 4.0 : f32
    memref.store %ad0, %adiv[%c0] : memref<3xf32>
    memref.store %ad1, %adiv[%i1] : memref<3xf32>
    memref.store %ad2, %adiv[%i2] : memref<3xf32>

    %A = memref.alloc() : memref<256x4096xbf16>
    scf.for %i = %c0 to %c256 step %c1 {
      scf.for %k = %c0 to %c4K step %c1 {
        %t = arith.divui %k, %c32 : index
        %fam = arith.remui %t, %c3 : index
        %ik = arith.addi %i, %k : index
        %idx = arith.remui %ik, %c8 : index
        %v = memref.load %lut[%idx] : memref<8xf32>
        %d = memref.load %adiv[%fam] : memref<3xf32>
        %a = arith.divf %v, %d : f32
        %ab = arith.truncf %a : f32 to bf16
        memref.store %ab, %A[%i, %k] : memref<256x4096xbf16>
        memref.store %a, %A_f32[%i, %k] : memref<256x4096xf32>
      }
    }



    %c16i8 = arith.constant 16 : i8

    // Byte [m, j] of B holds K elements 2m and 2m+1 of column j, both in the
    // same block of 32, so they share one scale.
    %B = memref.alloc() : memref<2048x256xi8>
    %B_scale = memref.alloc() : memref<128x256xf8E8M0FNU>
    scf.for %m = %c0 to %c2K step %c1 {
      %k0 = arith.muli %m, %c2 : index
      %k1 = arith.addi %k0, %c1 : index
      %t = arith.divui %k0, %c32 : index
      scf.for %j = %c0 to %c256 step %c1 {
        %sa = arith.addi %j, %k0 : index
        %sb = arith.addi %j, %k1 : index
        %idx0 = arith.remui %sa, %c8 : index
        %idx1 = arith.remui %sb, %c8 : index
        %lo = arith.index_cast %idx0 : index to i8
        %hi = arith.index_cast %idx1 : index to i8
        %hi4 = arith.muli %hi, %c16i8 : i8
        %byte = arith.ori %lo, %hi4 : i8
        memref.store %byte, %B[%m, %j] : memref<2048x256xi8>
        %ts = arith.addi %t, %j : index
        %sidx = arith.remui %ts, %c3 : index
        %sv = memref.load %scf32[%sidx] : memref<3xf32>
        %v0 = memref.load %lut[%idx0] : memref<8xf32>
        %v1 = memref.load %lut[%idx1] : memref<8xf32>
        %p0 = arith.mulf %v0, %sv : f32
        %p1 = arith.mulf %v1, %sv : f32
        memref.store %p0, %B_f32[%k0, %j] : memref<4096x256xf32>
        memref.store %p1, %B_f32[%k1, %j] : memref<4096x256xf32>
      }
    }
    scf.for %t = %c0 to %c128 step %c1 {
      scf.for %j = %c0 to %c256 step %c1 {
        %ts = arith.addi %t, %j : index
        %sidx = arith.remui %ts, %c3 : index
        %se = memref.load %sc[%sidx] : memref<3xf8E8M0FNU>
        memref.store %se, %B_scale[%t, %j] : memref<128x256xf8E8M0FNU>
      }
    }

    %C = memref.alloc() : memref<256x256xf32>
    scf.for %i = %c0 to %c256 step %c1 {
      scf.for %j = %c0 to %c256 step %c1 {
        memref.store %c0f32, %C[%i, %j] : memref<256x256xf32>
      }
    }



    // Reference GEMM on the host over the f32 shadows: A as given, B with its
    // block scale folded in. A is a multiple of 0.25 bounded by 6 and B with
    // its scale is a multiple of 0.25 bounded by 12, so every product is a
    // multiple of 0.0625 and the largest sum needs well under 2^24 units of it.
    // No addition rounds, so the result does not depend on the order or
    // internal precision the hardware picks, which the MX spec leaves
    // implementation defined.
    %C_ref = memref.alloc() : memref<256x256xf32>
    call @gemm_ref(%A_f32, %B_f32, %C_ref) : (memref<256x4096xf32>, memref<4096x256xf32>, memref<256x256xf32>) -> ()

    %C_res = call @test(%A, %B, %B_scale, %C) : (memref<256x4096xbf16>, memref<2048x256xi8>, memref<128x256xf8E8M0FNU>, memref<256x256xf32>) -> memref<256x256xf32>
    %C_cast = memref.cast %C_res : memref<256x256xf32> to memref<*xf32>
    %C_ref_cast = memref.cast %C_ref : memref<256x256xf32> to memref<*xf32>
    %diff = call @verifyMemRefF32(%C_cast, %C_ref_cast) : (memref<*xf32>, memref<*xf32>) -> i64
    %prefix = llvm.mlir.addressof @mismatches_str : !llvm.ptr
    llvm.call @printString(%prefix) : (!llvm.ptr) -> ()
    call @printI64(%diff) : (i64) -> ()
    call @printNewline() : () -> ()
    //call @printMemrefF32(%C_cast) : (memref<*xf32>) -> ()

    // MISMATCH: {{^mismatches: 0$}}
    memref.dealloc %A_f32 : memref<256x4096xf32>
    memref.dealloc %B_f32 : memref<4096x256xf32>
    memref.dealloc %lut : memref<8xf32>
    memref.dealloc %adiv : memref<3xf32>
    memref.dealloc %sc : memref<3xf8E8M0FNU>
    memref.dealloc %scf32 : memref<3xf32>
    memref.dealloc %A : memref<256x4096xbf16>
    memref.dealloc %B : memref<2048x256xi8>
    memref.dealloc %B_scale : memref<128x256xf8E8M0FNU>
    memref.dealloc %C : memref<256x256xf32>
    memref.dealloc %C_res : memref<256x256xf32>
    return
  }
  func.func private @verifyMemRefF32(%acutal : memref<*xf32>, %expected : memref<*xf32>) -> i64 attributes { llvm.emit_c_interface }
  func.func private @printI64(%num : i64)
  func.func private @printNewline()

  // Print the mismatch count as "mismatches: <n>" rather than bare, so the
  // check cannot be satisfied by an unrelated 0: the runtime is free to write
  // diagnostics to stdout, and a bare "0" check matches a 0 anywhere in them,
  // including inside a larger number.
  llvm.mlir.global internal constant @mismatches_str("mismatches: \00")
  llvm.func @printString(!llvm.ptr)
  //func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes { llvm.emit_c_interface }


  // Plain host GEMM, used to build the expected result.
  func.func @gemm_ref(%A: memref<256x4096xf32>, %B: memref<4096x256xf32>,
                      %C: memref<256x256xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c4K = arith.constant 4096 : index
    %zero = arith.constant 0.0 : f32
    scf.for %i = %c0 to %c256 step %c1 {
      scf.for %j = %c0 to %c256 step %c1 {
        %acc = scf.for %k = %c0 to %c4K step %c1
            iter_args(%sum = %zero) -> (f32) {
          %a = memref.load %A[%i, %k] : memref<256x4096xf32>
          %b = memref.load %B[%k, %j] : memref<4096x256xf32>
          %p = arith.mulf %a, %b : f32
          %s = arith.addf %sum, %p : f32
          scf.yield %s : f32
        }
        memref.store %acc, %C[%i, %j] : memref<256x256xf32>
      }
    }
    return
  }

}