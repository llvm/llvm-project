// RUN: mlir-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup zebin-chip=cri"
// RUN-DISABLED: mlir-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup zebin-chip=cri" \
// RUN-DISABLED: | mlir-runner \
// RUN-DISABLED:   --shared-libs=%mlir_levelzero_runtime \
// RUN-DISABLED:   --shared-libs=%mlir_runner_utils \
// RUN-DISABLED:   --shared-libs=%mlir_c_runner_utils \
// RUN-DISABLED:   --entry-point-result=void \
// RUN-DISABLED: | FileCheck --check-prefix=MISMATCH %s

// Operands, as they arrive and as the kernel uses them:
//   A        packed fp4, two values per byte along K: it crosses as
//            memref<524288xi8> and is viewed as memref<256x4096xf4E2M1FN> on
//            the device (see @test for why the bytes cross flat).
//   B        pre-packed as memref<2048x256xi8>, byte [t, j] holding K elements
//            2t and 2t+1 of column j. The kernel bitcasts each byte into two
//            fp4 along N, then deinterleaves, interleaves and transposes to
//            rebuild the 1024x32 K-major operand dpas_mx takes.
//   scales   memref<256x128xf8E8M0FNU> for A and memref<128x256xf8E8M0FNU> for
//            B, one f8E8M0 scale per 32 elements of K.
//   C        f32, a 32x32 workgroup tile with sg_layout [2, 2].

// Note: layouts used by dpas_mx need to match HW constaint. Otherwise dpas_mx is not unrolled.
#a = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 1024], inst_data = [8, 64], lane_layout = [1, 16], lane_data = [1, 4]>
#b_packed = #xegpu.layout<sg_layout = [2, 2], sg_data = [512, 16], inst_data = [32, 16], lane_layout = [1, 16], lane_data = [4, 1]>
#b = #xegpu.layout<sg_layout = [2, 2], sg_data = [1024, 16], inst_data = [64, 16], lane_layout = [1, 16], lane_data = [8, 1]>
#c = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 16], inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>
// Note: inst_data is chosen to utilize 2D block load
#a_scale = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 32], inst_data = [16, 32], lane_layout = [16, 1], lane_data = [1, 1]>
#b_scale = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 16], inst_data = [32, 16], lane_layout = [1, 16], lane_data = [1, 1]>
// Note: scales for dpas_mx needs separate layouts with inst_data to match HW constraint. Otherwise dpas_mx is not unrolled
#dpas_a_scale = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 32], inst_data = [8, 2], lane_layout = [8, 1], lane_data = [1, 1]>
#dpas_b_scale = #xegpu.layout<sg_layout = [2, 2], sg_data = [32, 16], inst_data = [2, 16], lane_layout = [1, 16], lane_data = [1, 1]>


module @gemm attributes {gpu.container_module} {
  gpu.module @kernel {
    gpu.func @gemm_mxfp(%arg0: memref<256x4096xf4E2M1FN>, %arg1: memref<2048x256xi8>, %arg2: memref<256x128xf8E8M0FNU>, %arg3: memref<128x256xf8E8M0FNU>, %arg4: memref<256x256xf32>) kernel {
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

      %a_tdesc = xegpu.create_nd_tdesc %arg0 : memref<256x4096xf4E2M1FN> -> !xegpu.tensor_desc<32x1024xf4E2M1FN>
      %bp_tdesc = xegpu.create_nd_tdesc %arg1 : memref<2048x256xi8> -> !xegpu.tensor_desc<512x32xi8>
      %a_scale_tdesc = xegpu.create_nd_tdesc %arg2 : memref<256x128xf8E8M0FNU> -> !xegpu.tensor_desc<32x32xf8E8M0FNU>
      %b_scale_tdesc = xegpu.create_nd_tdesc %arg3 : memref<128x256xf8E8M0FNU> -> !xegpu.tensor_desc<32x32xf8E8M0FNU>

      // Load initial C
      %cd_tdesc = xegpu.create_nd_tdesc %arg4 : memref<256x256xf32> -> !xegpu.tensor_desc<32x32xf32, #c>
      %c_init = xegpu.load_nd %cd_tdesc[%m, %n] <{layout = #c}>: !xegpu.tensor_desc<32x32xf32, #c> -> vector<32x32xf32>

      %res:3 = scf.for %k = %c0 to %kbound step %kstep
        iter_args(%c_partial = %c_init, %kb = %c0, %kscale = %c0) -> (vector<32x32xf32>, index, index) {
        // load_nd with offset
        %a = xegpu.load_nd %a_tdesc[%m, %k] <{layout = #a}>: !xegpu.tensor_desc<32x1024xf4E2M1FN> -> vector<32x1024xf4E2M1FN>
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

        %scale_a = xegpu.load_nd %a_scale_tdesc[%m, %kscale] <{layout = #a_scale}>: !xegpu.tensor_desc<32x32xf8E8M0FNU> -> vector<32x32xf8E8M0FNU>

        %scale_b = xegpu.load_nd %b_scale_tdesc[%kscale, %n] <{layout = #b_scale}>: !xegpu.tensor_desc<32x32xf8E8M0FNU> -> vector<32x32xf8E8M0FNU>
        %new_c_partial = xegpu.dpas_mx %a, %b, %c_partial scale_a = %scale_a scale_b = %scale_b
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

  // A crosses the runtime boundary as a flat i8 buffer rather than as a 4-bit
  // memref. gpu.alloc/memcpy/dealloc size a memref by rounding the element type
  // up to a whole byte - the size is lowered as gep(null, numElements) on i4,
  // which LLVM strides by 1 byte - so a packed fp4 memref is sized at twice its
  // real footprint. The copy would then read 512 KB past this 524288-byte host
  // buffer; because mgpuMemcpy is asynchronous the fault surfaces later, at the
  // stream sync inside the first mgpuMemFree. Handing over the packed bytes and
  // viewing them as fp4 on the device keeps every runtime size exact, and is
  // what the B operand already does.
  func.func @test(%a_flat: memref<524288xi8>, %b: memref<2048x256xi8>, %a_scale: memref<256x128xf8E8M0FNU>, %b_scale: memref<128x256xf8E8M0FNU>, %c: memref<256x256xf32>) -> memref<256x256xf32> attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index

    %memref_a_flat = gpu.alloc() : memref<524288xi8>
    gpu.memcpy %memref_a_flat, %a_flat : memref<524288xi8>, memref<524288xi8>
    %memref_a = memref.view %memref_a_flat[%c0][] : memref<524288xi8> to memref<256x4096xf4E2M1FN>

    %memref_b = gpu.alloc() : memref<2048x256xi8>
    gpu.memcpy %memref_b, %b : memref<2048x256xi8>, memref<2048x256xi8>

    %memref_c = gpu.alloc() : memref<256x256xf32>
    gpu.memcpy %memref_c, %c : memref<256x256xf32>, memref<256x256xf32>

    %memref_a_scale = gpu.alloc() : memref<256x128xf8E8M0FNU>
    gpu.memcpy %memref_a_scale, %a_scale : memref<256x128xf8E8M0FNU>, memref<256x128xf8E8M0FNU>

    %memref_b_scale = gpu.alloc() : memref<128x256xf8E8M0FNU>
    gpu.memcpy %memref_b_scale, %b_scale : memref<128x256xf8E8M0FNU>, memref<128x256xf8E8M0FNU>

    gpu.launch_func @kernel::@gemm_mxfp blocks in (%c8, %c8, %c1) threads in (%c64, %c1, %c1)
    args(%memref_a : memref<256x4096xf4E2M1FN>, %memref_b : memref<2048x256xi8>, %memref_a_scale : memref<256x128xf8E8M0FNU>, %memref_b_scale : memref<128x256xf8E8M0FNU>, %memref_c : memref<256x256xf32>)
    gpu.dealloc %memref_a_flat : memref<524288xi8>
    gpu.dealloc %memref_b : memref<2048x256xi8>
    gpu.dealloc %memref_a_scale : memref<256x128xf8E8M0FNU>
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
    %c0f32 = arith.constant 0.0 : f32

    // The 8 magnitudes e2m1 can represent, indexed by their e2m1 bit pattern:
    // code c encodes lut[c], so a packed byte holding two copies of code c is
    // c * 0x11. Every one of these is also exact in f8E5M2, f8E4M3FN, bf16 and
    // f32, so the fp4, fp8 and bf16 tests can share one input set and one
    // reference result.
    %lut = memref.alloc() : memref<8xf32>
    %l0 = arith.constant 0.0 : f32
    %l1 = arith.constant 0.5 : f32
    %l2 = arith.constant 1.0 : f32
    %l3 = arith.constant 1.5 : f32
    %l4 = arith.constant 2.0 : f32
    %l5 = arith.constant 3.0 : f32
    %l6 = arith.constant 4.0 : f32
    %l7 = arith.constant 6.0 : f32
    memref.store %l0, %lut[%c0] : memref<8xf32>
    %i1 = arith.constant 1 : index
    memref.store %l1, %lut[%i1] : memref<8xf32>
    %i2 = arith.constant 2 : index
    memref.store %l2, %lut[%i2] : memref<8xf32>
    %i3 = arith.constant 3 : index
    memref.store %l3, %lut[%i3] : memref<8xf32>
    %i4 = arith.constant 4 : index
    memref.store %l4, %lut[%i4] : memref<8xf32>
    %i5 = arith.constant 5 : index
    memref.store %l5, %lut[%i5] : memref<8xf32>
    %i6 = arith.constant 6 : index
    memref.store %l6, %lut[%i6] : memref<8xf32>
    %i7 = arith.constant 7 : index
    memref.store %l7, %lut[%i7] : memref<8xf32>

    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %c2048 = arith.constant 2048 : index
    %c16i8 = arith.constant 16 : i8

    // A's per-K-block divisor and B's per-K-block scale, three of each and all
    // powers of two. The packed operands hold the unscaled codes; the scales
    // passed to dpas_mx reproduce the shared input set.
    %adiv = memref.alloc() : memref<3xf32>
    %ainv = memref.alloc() : memref<3xf8E8M0FNU>
    %bscf = memref.alloc() : memref<3xf32>
    %bsce = memref.alloc() : memref<3xf8E8M0FNU>
    %ad0 = arith.constant 1.0 : f32
    %ad1 = arith.constant 2.0 : f32
    %ad2 = arith.constant 4.0 : f32
    memref.store %ad0, %adiv[%c0] : memref<3xf32>
    memref.store %ad1, %adiv[%i1] : memref<3xf32>
    memref.store %ad2, %adiv[%i2] : memref<3xf32>
    %ai0 = arith.constant 1.0 : f8E8M0FNU
    %ai1 = arith.constant 0.5 : f8E8M0FNU
    %ai2 = arith.constant 0.25 : f8E8M0FNU
    memref.store %ai0, %ainv[%c0] : memref<3xf8E8M0FNU>
    memref.store %ai1, %ainv[%i1] : memref<3xf8E8M0FNU>
    memref.store %ai2, %ainv[%i2] : memref<3xf8E8M0FNU>
    %bs0 = arith.constant 0.5 : f32
    %bs1 = arith.constant 1.0 : f32
    %bs2 = arith.constant 2.0 : f32
    memref.store %bs0, %bscf[%c0] : memref<3xf32>
    memref.store %bs1, %bscf[%i1] : memref<3xf32>
    memref.store %bs2, %bscf[%i2] : memref<3xf32>
    %be0 = arith.constant 0.5 : f8E8M0FNU
    %be1 = arith.constant 1.0 : f8E8M0FNU
    %be2 = arith.constant 2.0 : f8E8M0FNU
    memref.store %be0, %bsce[%c0] : memref<3xf8E8M0FNU>
    memref.store %be1, %bsce[%i1] : memref<3xf8E8M0FNU>
    memref.store %be2, %bsce[%i2] : memref<3xf8E8M0FNU>
    %c3 = arith.constant 3 : index
    %c32 = arith.constant 32 : index

    // f32 shadows of A and B, filled from the same loop that writes the device
    // operands, so the reference cannot drift from what the kernel is given.
    %A_f32 = memref.alloc() : memref<256x4096xf32>
    %B_f32 = memref.alloc() : memref<4096x256xf32>

    // A is row major fp4, two values per byte along K, so byte m of row i holds
    // K elements 2m and 2m+1. The low nibble is assumed to hold the lower K.
    // Only the packed bytes are needed here: @test hands them to the device and
    // views them as fp4 there, so no fp4 memref crosses the runtime boundary.
    %A_flatbytes = memref.alloc() : memref<524288xi8>
    scf.for %i = %c0 to %c256 step %c1 {
      %row = arith.muli %i, %c2048 : index
      scf.for %m = %c0 to %c2048 step %c1 {
        %k0 = arith.muli %m, %c2 : index
        %k1 = arith.addi %k0, %c1 : index
        %s0 = arith.addi %i, %k0 : index
        %s1 = arith.addi %i, %k1 : index
        %idx0 = arith.remui %s0, %c8 : index
        %idx1 = arith.remui %s1, %c8 : index
        %lo = arith.index_cast %idx0 : index to i8
        %hi = arith.index_cast %idx1 : index to i8
        %hi4 = arith.muli %hi, %c16i8 : i8
        %byte = arith.ori %lo, %hi4 : i8
        %pos = arith.addi %row, %m : index
        memref.store %byte, %A_flatbytes[%pos] : memref<524288xi8>
        %t = arith.divui %k0, %c32 : index
        %fam = arith.remui %t, %c3 : index
        %d = memref.load %adiv[%fam] : memref<3xf32>
        %v0 = memref.load %lut[%idx0] : memref<8xf32>
        %v1 = memref.load %lut[%idx1] : memref<8xf32>
        %a0 = arith.divf %v0, %d : f32
        %a1 = arith.divf %v1, %d : f32
        memref.store %a0, %A_f32[%i, %k0] : memref<256x4096xf32>
        memref.store %a1, %A_f32[%i, %k1] : memref<256x4096xf32>
      }
    }

    // Byte [t, x] of B holds K elements 2t and 2t+1 of column x: the kernel
    // bitcasts each byte into two fp4 along N, then deinterleaves and
    // transposes to put them back along K.
    %B = memref.alloc() : memref<2048x256xi8>
    scf.for %m = %c0 to %c2K step %c1 {
      scf.for %j = %c0 to %c256 step %c1 {
        %k0 = arith.muli %m, %c2 : index
        %k1 = arith.addi %k0, %c1 : index
        %s0 = arith.addi %j, %k0 : index
        %s1 = arith.addi %j, %k1 : index
        %idx0 = arith.remui %s0, %c8 : index
        %idx1 = arith.remui %s1, %c8 : index
        %lo = arith.index_cast %idx0 : index to i8
        %hi = arith.index_cast %idx1 : index to i8
        %hi4 = arith.muli %hi, %c16i8 : i8
        %byte = arith.ori %lo, %hi4 : i8
        memref.store %byte, %B[%m, %j] : memref<2048x256xi8>
        %t = arith.divui %k0, %c32 : index
        %ts = arith.addi %t, %j : index
        %sidx = arith.remui %ts, %c3 : index
        %sv = memref.load %bscf[%sidx] : memref<3xf32>
        %v0 = memref.load %lut[%idx0] : memref<8xf32>
        %v1 = memref.load %lut[%idx1] : memref<8xf32>
        %b0 = arith.mulf %v0, %sv : f32
        %b1 = arith.mulf %v1, %sv : f32
        memref.store %b0, %B_f32[%k0, %j] : memref<4096x256xf32>
        memref.store %b1, %B_f32[%k1, %j] : memref<4096x256xf32>
      }
    }

    %C = memref.alloc() : memref<256x256xf32>
    scf.for %i = %c0 to %c256 step %c1 {
      scf.for %j = %c0 to %c256 step %c1 {
        memref.store %c0f32, %C[%i, %j] : memref<256x256xf32>
      }
    }

    %A_scale = memref.alloc() : memref<256x128xf8E8M0FNU>
    scf.for %i = %c0 to %c256 step %c1 {
      scf.for %t = %c0 to %c128 step %c1 {
        %fam = arith.remui %t, %c3 : index
        %sv = memref.load %ainv[%fam] : memref<3xf8E8M0FNU>
        memref.store %sv, %A_scale[%i, %t] : memref<256x128xf8E8M0FNU>
      }
    }

    %B_scale = memref.alloc() : memref<128x256xf8E8M0FNU>
    scf.for %t = %c0 to %c128 step %c1 {
      scf.for %j = %c0 to %c256 step %c1 {
        %ts = arith.addi %t, %j : index
        %sidx = arith.remui %ts, %c3 : index
        %sv = memref.load %bsce[%sidx] : memref<3xf8E8M0FNU>
        memref.store %sv, %B_scale[%t, %j] : memref<128x256xf8E8M0FNU>
      }
    }


    // Reference GEMM on the host. Every product is a multiple of 0.25 and the
    // largest result is well under 2^24, so the f32 accumulation is exact and
    // independent of summation order: the device result has to match bit for
    // bit.
    //
    // Consecutive K values differ, so the two fp4 values sharing a byte differ
    // and the packing is exercised. A and B are packed the same way, and
    // swapping the nibbles of both would only reorder the two products inside
    // one sum, so the expected result does not depend on which nibble holds the
    // lower K index. Swapping just one of them does change the result, and was
    // checked to be caught.
    %C_ref = memref.alloc() : memref<256x256xf32>
    call @gemm_ref(%A_f32, %B_f32, %C_ref) : (memref<256x4096xf32>, memref<4096x256xf32>, memref<256x256xf32>) -> ()

    %C_res = call @test(%A_flatbytes, %B, %A_scale, %B_scale, %C) : (memref<524288xi8>, memref<2048x256xi8>, memref<256x128xf8E8M0FNU>, memref<128x256xf8E8M0FNU>, memref<256x256xf32>) -> memref<256x256xf32>
    %C_cast = memref.cast %C_res : memref<256x256xf32> to memref<*xf32>
    %C_ref_cast = memref.cast %C_ref : memref<256x256xf32> to memref<*xf32>
    %diff = call @verifyMemRefF32(%C_cast, %C_ref_cast) : (memref<*xf32>, memref<*xf32>) -> i64
    %prefix = llvm.mlir.addressof @mismatches_str : !llvm.ptr
    llvm.call @printString(%prefix) : (!llvm.ptr) -> ()
    call @printI64(%diff) : (i64) -> ()
    call @printNewline() : () -> ()
    //call @printMemrefF32(%C_cast) : (memref<*xf32>) -> ()

    // MISMATCH: {{^mismatches: 0$}}
    memref.dealloc %A_flatbytes : memref<524288xi8>
    memref.dealloc %A_f32 : memref<256x4096xf32>
    memref.dealloc %B_f32 : memref<4096x256xf32>
    memref.dealloc %lut : memref<8xf32>
    memref.dealloc %adiv : memref<3xf32>
    memref.dealloc %ainv : memref<3xf8E8M0FNU>
    memref.dealloc %bscf : memref<3xf32>
    memref.dealloc %bsce : memref<3xf8E8M0FNU>
    memref.dealloc %C_ref : memref<256x256xf32>
    memref.dealloc %B : memref<2048x256xi8>
    memref.dealloc %A_scale : memref<256x128xf8E8M0FNU>
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
  //func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes { llvm.emit_c_interface }

}
