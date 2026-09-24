// RUN: mlir-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup zebin-chip=cri"
// RUN-DISABLED: mlir-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup zebin-chip=cri" \
// RUN-DISABLED: | mlir-runner \
// RUN-DISABLED:   --shared-libs=%mlir_levelzero_runtime \
// RUN-DISABLED:   --shared-libs=%mlir_runner_utils \
// RUN-DISABLED:   --shared-libs=%mlir_c_runner_utils \
// RUN-DISABLED:   --entry-point-result=void \
// RUN-DISABLED: | FileCheck --check-prefix=MISMATCH %s

// mx-fp8 variant of simple_mxfp_gemm_F4.mlir: A and B are pre-quantized to
// f8E5M2 with one f8E8M0 scale per 32 elements along K, the same block size
// the fp4 version uses. Unlike fp4, an element is a whole byte, so B needs no
// packing and is loaded directly.
//
// A dpas_mx instruction takes A as 8x32 and B as 32x16 for fp8, half the K of
// the fp4 case, so one scale covers exactly one instruction along K and the
// dpas scale inst_data is [8, 1] / [1, 16] rather than [8, 2] / [2, 16].
//
// The K step is 512 rather than the 1024 the fp4 version uses, so that a
// per-subgroup operand tile is the same 8 KB at 8 bits per element as it is at
// 4. Keeping fp4's step would double the tile and spill heavily, which would
// make any measurement of this kernel be about spilling rather than dpas_mx.

// Operands, as they arrive and as the kernel uses them:
//   A        memref<256x4096xf8E5M2>, loaded directly; no packing.
//   B        memref<4096x256xf8E5M2>, loaded directly; no packing.
//   scales   memref<256x128xf8E8M0FNU> for A and memref<128x256xf8E8M0FNU> for
//            B, one f8E8M0 scale per 32 elements of K.
//   C        f32, a 32x32 workgroup tile with sg_layout [2, 2].

// Note: layouts used by dpas_mx need to match HW constaint. Otherwise dpas_mx is not unrolled.
#a = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 512], inst_data = [8, 32], lane_layout = [1, 16], lane_data = [1, 2]>
#b = #xegpu.layout<sg_layout = [2, 2], sg_data = [512, 16], inst_data = [32, 16], lane_layout = [1, 16], lane_data = [4, 1]>
#c = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 16], inst_data = [8, 16], lane_layout = [1, 16], lane_data = [1, 1]>
// Note: inst_data is chosen to utilize 2D block load
#a_scale = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 16], inst_data = [16, 16], lane_layout = [16, 1], lane_data = [1, 1]>
#b_scale = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 16], inst_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>
// Note: scales for dpas_mx needs separate layouts with inst_data to match HW constraint. Otherwise dpas_mx is not unrolled
#dpas_a_scale = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 16], inst_data = [8, 1], lane_layout = [8, 1], lane_data = [1, 1]>
#dpas_b_scale = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 16], inst_data = [1, 16], lane_layout = [1, 16], lane_data = [1, 1]>


module @gemm attributes {gpu.container_module} {
  gpu.module @kernel {
    gpu.func @gemm_mxfp(%arg0: memref<256x4096xf8E5M2>, %arg1: memref<4096x256xf8E5M2>, %arg2: memref<256x128xf8E8M0FNU>, %arg3: memref<128x256xf8E8M0FNU>, %arg4: memref<256x256xf32>) kernel {
      %c0 = arith.constant 0 : index
      %mstep = arith.constant 32 : index
      %nstep = arith.constant 32 : index
      %kstep = arith.constant 512 : index
      %kbound = arith.constant 4096 : index
      %kscalestep = arith.constant 16 : index
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %m = arith.muli %block_id_x, %mstep : index
      %n = arith.muli %block_id_y, %nstep : index

      %a_tdesc = xegpu.create_nd_tdesc %arg0 : memref<256x4096xf8E5M2> -> !xegpu.tensor_desc<32x512xf8E5M2>
      %b_tdesc = xegpu.create_nd_tdesc %arg1 : memref<4096x256xf8E5M2> -> !xegpu.tensor_desc<512x32xf8E5M2>
      %a_scale_tdesc = xegpu.create_nd_tdesc %arg2 : memref<256x128xf8E8M0FNU> -> !xegpu.tensor_desc<32x16xf8E8M0FNU>
      %b_scale_tdesc = xegpu.create_nd_tdesc %arg3 : memref<128x256xf8E8M0FNU> -> !xegpu.tensor_desc<16x32xf8E8M0FNU>

      // Load initial C
      %cd_tdesc = xegpu.create_nd_tdesc %arg4 : memref<256x256xf32> -> !xegpu.tensor_desc<32x32xf32, #c>
      %c_init = xegpu.load_nd %cd_tdesc[%m, %n] <{layout = #c}>: !xegpu.tensor_desc<32x32xf32, #c> -> vector<32x32xf32>

      %res:2 = scf.for %k = %c0 to %kbound step %kstep
        iter_args(%c_partial = %c_init, %kscale = %c0) -> (vector<32x32xf32>, index) {
        // A and B are already in mx-fp8, so both are loaded directly. B is
        // indexed by %k as well, since an fp8 element occupies a whole byte
        // and the K dimension is not packed.
        %a = xegpu.load_nd %a_tdesc[%m, %k] <{layout = #a}>: !xegpu.tensor_desc<32x512xf8E5M2> -> vector<32x512xf8E5M2>
        %b = xegpu.load_nd %b_tdesc[%k, %n] <{layout = #b}>: !xegpu.tensor_desc<512x32xf8E5M2> -> vector<512x32xf8E5M2>

        // One scale per 32 elements along K, so a 512 wide K chunk needs 16
        // scales per row of A and 16 per column of B.
        %scale_a = xegpu.load_nd %a_scale_tdesc[%m, %kscale] <{layout = #a_scale}>: !xegpu.tensor_desc<32x16xf8E8M0FNU> -> vector<32x16xf8E8M0FNU>
        %scale_b = xegpu.load_nd %b_scale_tdesc[%kscale, %n] <{layout = #b_scale}>: !xegpu.tensor_desc<16x32xf8E8M0FNU> -> vector<16x32xf8E8M0FNU>

        %new_c_partial = xegpu.dpas_mx %a, %b, %c_partial scale_a = %scale_a scale_b = %scale_b
              <{layout_a = #a,
               layout_b = #b,
               layout_cd = #c,
               layout_a_scale = #dpas_a_scale,
               layout_b_scale = #dpas_b_scale}>
            : (vector<32x512xf8E5M2>, vector<512x32xf8E5M2>,
               vector<32x32xf32>,
               vector<32x16xf8E8M0FNU>, vector<16x32xf8E8M0FNU>)
            -> vector<32x32xf32>

        // The scale tiles take a different step compared to a and b.
        %new_kscale = arith.addi %kscale, %kscalestep : index
        scf.yield %new_c_partial, %new_kscale : vector<32x32xf32>, index
      }

      // store_nd with offset
      xegpu.store_nd %res#0, %cd_tdesc[%m, %n] <{layout = #c}> : vector<32x32xf32>, !xegpu.tensor_desc<32x32xf32, #c>
      gpu.return
    }
  }

  func.func @test(%a: memref<256x4096xf8E5M2>, %b: memref<4096x256xf8E5M2>, %a_scale: memref<256x128xf8E8M0FNU>, %b_scale: memref<128x256xf8E8M0FNU>, %c: memref<256x256xf32>) -> memref<256x256xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index

    %memref_a = gpu.alloc() : memref<256x4096xf8E5M2>
    gpu.memcpy %memref_a, %a : memref<256x4096xf8E5M2>, memref<256x4096xf8E5M2>

    %memref_b = gpu.alloc() : memref<4096x256xf8E5M2>
    gpu.memcpy %memref_b, %b : memref<4096x256xf8E5M2>, memref<4096x256xf8E5M2>

    %memref_c = gpu.alloc() : memref<256x256xf32>
    gpu.memcpy %memref_c, %c : memref<256x256xf32>, memref<256x256xf32>

    %memref_a_scale = gpu.alloc() : memref<256x128xf8E8M0FNU>
    gpu.memcpy %memref_a_scale, %a_scale : memref<256x128xf8E8M0FNU>, memref<256x128xf8E8M0FNU>

    %memref_b_scale = gpu.alloc() : memref<128x256xf8E8M0FNU>
    gpu.memcpy %memref_b_scale, %b_scale : memref<128x256xf8E8M0FNU>, memref<128x256xf8E8M0FNU>

    gpu.launch_func @kernel::@gemm_mxfp blocks in (%c8, %c8, %c1) threads in (%c64, %c1, %c1)
    args(%memref_a : memref<256x4096xf8E5M2>, %memref_b : memref<4096x256xf8E5M2>, %memref_a_scale : memref<256x128xf8E8M0FNU>, %memref_b_scale : memref<128x256xf8E8M0FNU>, %memref_c : memref<256x256xf32>)
    gpu.dealloc %memref_a : memref<256x4096xf8E5M2>
    gpu.dealloc %memref_b : memref<4096x256xf8E5M2>
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
    %c4K = arith.constant 4096 : index
    %c0f32 = arith.constant 0.0 : f32

    // The 8 magnitudes e2m1 can represent. They are exact in f8E5M2, bf16 and
    // f32 as well, so the fp4, fp8 and bf16 tests share one input set and one
    // reference result.
    %lut = memref.alloc() : memref<8xf32>
    %lut8 = memref.alloc() : memref<8xf8E5M2>
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
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
    %e0 = arith.constant 0.0 : f8E5M2
    %e1 = arith.constant 0.5 : f8E5M2
    %e2 = arith.constant 1.0 : f8E5M2
    %e3 = arith.constant 1.5 : f8E5M2
    %e4 = arith.constant 2.0 : f8E5M2
    %e5 = arith.constant 3.0 : f8E5M2
    %e6 = arith.constant 4.0 : f8E5M2
    %e7 = arith.constant 6.0 : f8E5M2
    memref.store %e0, %lut8[%c0] : memref<8xf8E5M2>
    memref.store %e1, %lut8[%i1] : memref<8xf8E5M2>
    memref.store %e2, %lut8[%i2] : memref<8xf8E5M2>
    memref.store %e3, %lut8[%i3] : memref<8xf8E5M2>
    memref.store %e4, %lut8[%i4] : memref<8xf8E5M2>
    memref.store %e5, %lut8[%i5] : memref<8xf8E5M2>
    memref.store %e6, %lut8[%i6] : memref<8xf8E5M2>
    memref.store %e7, %lut8[%i7] : memref<8xf8E5M2>

    %c8 = arith.constant 8 : index

    // f32 shadows of A and B, filled from the same loop that writes the device
    // operands, so the reference cannot drift from what the kernel is given.
    %A_f32 = memref.alloc() : memref<256x4096xf32>
    %B_f32 = memref.alloc() : memref<4096x256xf32>

    // A's per-K-block divisor and B's per-K-block scale, three of each and all
    // powers of two. The stored operands hold the unscaled values; the scales
    // passed to dpas_mx reproduce the shared input set.
    %adiv = memref.alloc() : memref<3xf32>
    %ainv = memref.alloc() : memref<3xf8E8M0FNU>
    %bsc = memref.alloc() : memref<3xf32>
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
    memref.store %bs0, %bsc[%c0] : memref<3xf32>
    memref.store %bs1, %bsc[%i1] : memref<3xf32>
    memref.store %bs2, %bsc[%i2] : memref<3xf32>
    %be0 = arith.constant 0.5 : f8E8M0FNU
    %be1 = arith.constant 1.0 : f8E8M0FNU
    %be2 = arith.constant 2.0 : f8E8M0FNU
    memref.store %be0, %bsce[%c0] : memref<3xf8E8M0FNU>
    memref.store %be1, %bsce[%i1] : memref<3xf8E8M0FNU>
    memref.store %be2, %bsce[%i2] : memref<3xf8E8M0FNU>
    %c3 = arith.constant 3 : index
    %c32 = arith.constant 32 : index

    %A = memref.alloc() : memref<256x4096xf8E5M2>
    scf.for %i = %c0 to %c256 step %c1 {
      scf.for %k = %c0 to %c4K step %c1 {
        %t = arith.divui %k, %c32 : index
        %fam = arith.remui %t, %c3 : index
        %ik = arith.addi %i, %k : index
        %idx = arith.remui %ik, %c8 : index
        %v8 = memref.load %lut8[%idx] : memref<8xf8E5M2>
        %v = memref.load %lut[%idx] : memref<8xf32>
        %d = memref.load %adiv[%fam] : memref<3xf32>
        %a = arith.divf %v, %d : f32
        memref.store %v8, %A[%i, %k] : memref<256x4096xf8E5M2>
        memref.store %a, %A_f32[%i, %k] : memref<256x4096xf32>
      }
    }

    %B = memref.alloc() : memref<4096x256xf8E5M2>
    scf.for %k = %c0 to %c4K step %c1 {
      %t = arith.divui %k, %c32 : index
      scf.for %j = %c0 to %c256 step %c1 {
        %jk = arith.addi %j, %k : index
        %idx = arith.remui %jk, %c8 : index
        %v8 = memref.load %lut8[%idx] : memref<8xf8E5M2>
        %v = memref.load %lut[%idx] : memref<8xf32>
        %ts = arith.addi %t, %j : index
        %sidx = arith.remui %ts, %c3 : index
        %sv = memref.load %bsc[%sidx] : memref<3xf32>
        %b = arith.mulf %v, %sv : f32
        memref.store %v8, %B[%k, %j] : memref<4096x256xf8E5M2>
        memref.store %b, %B_f32[%k, %j] : memref<4096x256xf32>
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

    // Reference GEMM on the host, over the f32 shadows of the same operands.
    // Every product is a multiple of 0.25 and the largest result is well under
    // 2^24, so the f32 accumulation is exact and independent of summation
    // order: the device result has to match bit for bit.
    %C_ref = memref.alloc() : memref<256x256xf32>
    call @gemm_ref(%A_f32, %B_f32, %C_ref) : (memref<256x4096xf32>, memref<4096x256xf32>, memref<256x256xf32>) -> ()

    %C_res = call @test(%A, %B, %A_scale, %B_scale, %C) : (memref<256x4096xf8E5M2>, memref<4096x256xf8E5M2>, memref<256x128xf8E8M0FNU>, memref<128x256xf8E8M0FNU>, memref<256x256xf32>) -> memref<256x256xf32>
    %C_cast = memref.cast %C_res : memref<256x256xf32> to memref<*xf32>
    %C_ref_cast = memref.cast %C_ref : memref<256x256xf32> to memref<*xf32>
    %diff = call @verifyMemRefF32(%C_cast, %C_ref_cast) : (memref<*xf32>, memref<*xf32>) -> i64
    %prefix = llvm.mlir.addressof @mismatches_str : !llvm.ptr
    llvm.call @printString(%prefix) : (!llvm.ptr) -> ()
    call @printI64(%diff) : (i64) -> ()
    call @printNewline() : () -> ()
    //call @printMemrefF32(%C_cast) : (memref<*xf32>) -> ()

    // MISMATCH: {{^mismatches: 0$}}
    memref.dealloc %A : memref<256x4096xf8E5M2>
    memref.dealloc %B : memref<4096x256xf8E5M2>
    memref.dealloc %A_scale : memref<256x128xf8E8M0FNU>
    memref.dealloc %B_scale : memref<128x256xf8E8M0FNU>
    memref.dealloc %C : memref<256x256xf32>
    memref.dealloc %C_res : memref<256x256xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printI64(i64)
  func.func private @printNewline()

  // Print the mismatch count as "mismatches: <n>" rather than bare, so the
  // check cannot be satisfied by an unrelated 0: the runtime is free to write
  // diagnostics to stdout, and a bare "0" check matches a 0 anywhere in them,
  // including inside a larger number.
  llvm.mlir.global internal constant @mismatches_str("mismatches: \00")
  llvm.func @printString(!llvm.ptr)
  func.func private @verifyMemRefF32(memref<*xf32>, memref<*xf32>) -> i64 attributes {llvm.emit_c_interface}

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
