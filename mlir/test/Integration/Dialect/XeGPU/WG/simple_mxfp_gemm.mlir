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


module @gemm attributes {gpu.container_module} {
  gpu.module @kernel {
    gpu.func @gemm_mxfp(%arg0: memref<1024x4096xf4E2M1FN>, %arg1: memref<2048x1024xi8>, %arg2: memref<1024x128xf8E8M0FNU>, %arg3: memref<128x1024xf8E8M0FNU>, %arg4: memref<1024x1024xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c128 = arith.constant 128 : index
      %c1024 = arith.constant 1024 : index
      %k = arith.constant 4096 : index
      %c16 = arith.constant 16 : index
      %scale_block = arith.constant 32 : index
      %kStep = arith.constant 512 : index
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %0 = arith.muli %block_id_x, %c128 : index
      %1 = arith.muli %block_id_y, %c128 : index

      %a_tdesc = xegpu.create_nd_tdesc %arg0 : memref<1024x4096xf4E2M1FN> -> !xegpu.tensor_desc<128x512xf4E2M1FN>
      %bp_tdesc = xegpu.create_nd_tdesc %arg1 : memref<2048x1024xi8> -> !xegpu.tensor_desc<256x128xi8>
      %a_scale_tdesc = xegpu.create_nd_tdesc %arg2 : memref<1024x128xf8E8M0FNU> -> !xegpu.tensor_desc<128x16xf8E8M0FNU>
      %b_scale_tdesc = xegpu.create_nd_tdesc %arg3 : memref<128x1024xf8E8M0FNU> -> !xegpu.tensor_desc<16x128xf8E8M0FNU>

      // Load initial C
      %cd_tdesc = xegpu.create_nd_tdesc %arg4 : memref<1024x1024xf32> -> !xegpu.tensor_desc<128x128xf32, #c>
      %c = xegpu.load_nd %cd_tdesc[%0, %1] {layout = #c}: !xegpu.tensor_desc<128x128xf32, #c> -> vector<128x128xf32>

      %d = scf.for %id_k = %c0 to %k step %kStep
        iter_args(%c_value = %c) -> (vector<128x128xf32>) {

        // load_nd with offset
        %a = xegpu.load_nd %a_tdesc[%0, %id_k] {layout = #a}: !xegpu.tensor_desc<128x512xf4E2M1FN> -> vector<128x512xf4E2M1FN>
        %id_k_packed = arith.divsi %id_k, %c2 : index
        %bp = xegpu.load_nd %bp_tdesc[%id_k_packed, %1] {layout = #b_packed}: !xegpu.tensor_desc<256x128xi8> -> vector<256x128xi8>

        // Bitcast to fp4: 256x128 uint8 -> 256x256 fp4 (each uint8 holds 2 fp4 values)
        %b_bitcast = vector.bitcast %bp : vector<256x128xi8> to vector<256x256xf4E2M1FN>

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

        %id_scale = arith.divsi %id_k, %scale_block : index
        %scale_a = xegpu.load_nd %a_scale_tdesc[%0, %id_scale] {layout = #a_scale}: !xegpu.tensor_desc<128x16xf8E8M0FNU> -> vector<128x16xf8E8M0FNU>

        %scale_b = xegpu.load_nd %b_scale_tdesc[%id_scale, %1] {layout = #b_scale}: !xegpu.tensor_desc<16x128xf8E8M0FNU> -> vector<16x128xf8E8M0FNU>
        %new_c_value = xegpu.dpas_mx %a, %b, %c_value scale_a = %scale_a scale_b = %scale_b
              {layout_a = #a,
               layout_b = #b,
               layout_cd = #c,
               layout_a_scale = #a_scale,
               layout_b_scale = #b_scale}
            : (vector<128x512xf4E2M1FN>, vector<512x128xf4E2M1FN>,
               vector<128x128xf32>,
               vector<128x16xf8E8M0FNU>, vector<16x128xf8E8M0FNU>)
            -> vector<128x128xf32>

        scf.yield %new_c_value : vector<128x128xf32>
      }

      // store_nd with offset
      xegpu.store_nd %d, %cd_tdesc[%0, %1] {layout = #c} : vector<128x128xf32>, !xegpu.tensor_desc<128x128xf32, #c>
      gpu.return
    }
  }

  func.func @test(%a: memref<1024x4096xf4E2M1FN>, %b: memref<2048x1024xi8>, %a_scale: memref<1024x128xf8E8M0FNU>, %b_scale: memref<128x1024xf8E8M0FNU>, %c: memref<1024x1024xf32>) -> memref<1024x1024xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index

    %memref_a = gpu.alloc() : memref<1024x4096xf4E2M1FN>
    gpu.memcpy %memref_a, %a : memref<1024x4096xf4E2M1FN>, memref<1024x4096xf4E2M1FN>

    %memref_b = gpu.alloc() : memref<2048x1024xi8>
    gpu.memcpy %memref_b, %b : memref<2048x1024xi8>, memref<2048x1024xi8>

    %memref_c = gpu.alloc() : memref<1024x1024xf32>
    gpu.memcpy %memref_c, %c : memref<1024x1024xf32>, memref<1024x1024xf32>

    %memref_a_scale = gpu.alloc() : memref<1024x128xf8E8M0FNU>
    gpu.memcpy %memref_a_scale, %a_scale : memref<1024x128xf8E8M0FNU>, memref<1024x128xf8E8M0FNU>

    %memref_b_scale = gpu.alloc() : memref<128x1024xf8E8M0FNU>
    gpu.memcpy %memref_b_scale, %b_scale : memref<128x1024xf8E8M0FNU>, memref<128x1024xf8E8M0FNU>

    gpu.launch_func @kernel::@gemm_mxfp blocks in (%c8, %c8, %c1) threads in (%c16, %c1, %c1)
    args(%memref_a : memref<1024x4096xf4E2M1FN>, %memref_b : memref<2048x1024xi8>, %memref_a_scale : memref<1024x128xf8E8M0FNU>, %memref_b_scale : memref<128x1024xf8E8M0FNU>, %memref_c : memref<1024x1024xf32>)
    gpu.dealloc %memref_a : memref<1024x4096xf4E2M1FN>
    gpu.dealloc %memref_b : memref<2048x1024xi8>
    gpu.dealloc %memref_a_scale : memref<1024x128xf8E8M0FNU>
    gpu.dealloc %memref_b_scale : memref<128x1024xf8E8M0FNU>

    %res = memref.alloc() : memref<1024x1024xf32>
    gpu.memcpy %res, %memref_c : memref<1024x1024xf32>, memref<1024x1024xf32>
    gpu.dealloc %memref_c : memref<1024x1024xf32>
    return %res : memref<1024x1024xf32>
  }

  func.func @main() attributes {llvm.emit_c_interface} {

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c1K = arith.constant 1024 : index
    %c2K = arith.constant 2048 : index
    %c4K = arith.constant 4096 : index
    %c2M = arith.constant 2097152 : index
    %c1e2m1 = arith.constant 1.0 : f4E2M1FN
    %c1packed_e2m1 = arith.constant 0x22 : i8
    %c0f32 = arith.constant 0.0 : f32
    %c1f8E8M0FNU = arith.constant 1.0 : f8E8M0FNU

    %A_flatbytes = memref.alloc() : memref<2097152xi8>
    %A = memref.view %A_flatbytes[%c0][] : memref<2097152xi8> to memref<1024x4096xf4E2M1FN>
    scf.for %i = %c0 to %c2M step %c1 {
      memref.store %c1packed_e2m1, %A_flatbytes[%i] : memref<2097152xi8>
    }

    %B = memref.alloc() : memref<2048x1024xi8>
    scf.for %i = %c0 to %c2K step %c1 {
      scf.for %j = %c0 to %c1K step %c1 {
        memref.store %c1packed_e2m1, %B[%i, %j] : memref<2048x1024xi8>
      }
    }

    %C = memref.alloc() : memref<1024x1024xf32>
    scf.for %i = %c0 to %c1K step %c1 {
      scf.for %j = %c0 to %c1K step %c1 {
        memref.store %c0f32, %C[%i, %j] : memref<1024x1024xf32>
      }
    }

    %A_scale = memref.alloc() : memref<1024x128xf8E8M0FNU>
    scf.for %i = %c0 to %c1K step %c1 {
      scf.for %j = %c0 to %c128 step %c1 {
        memref.store %c1f8E8M0FNU, %A_scale[%i, %j] : memref<1024x128xf8E8M0FNU>
      }
    }

    %B_scale = memref.alloc() : memref<128x1024xf8E8M0FNU>
    scf.for %i = %c0 to %c128 step %c1 {
      scf.for %j = %c0 to %c1K step %c1 {
        memref.store %c1f8E8M0FNU, %B_scale[%i, %j] : memref<128x1024xf8E8M0FNU>
      }
    }


    %c4Kf = arith.constant 4096.0 : f32
    %C_ref = memref.alloc() : memref<1024x1024xf32>
    scf.for %i = %c0 to %c1K step %c1 {
      scf.for %j = %c0 to %c1K step %c1 {
        memref.store %c4Kf, %C_ref[%i, %j] : memref<1024x1024xf32>
      }
    }

    %C_res = call @test(%A, %B, %A_scale, %B_scale, %C) : (memref<1024x4096xf4E2M1FN>, memref<2048x1024xi8>, memref<1024x128xf8E8M0FNU>, memref<128x1024xf8E8M0FNU>, memref<1024x1024xf32>) -> memref<1024x1024xf32>
    %C_cast = memref.cast %C_res : memref<1024x1024xf32> to memref<*xf32>
    %C_ref_cast = memref.cast %C_ref : memref<1024x1024xf32> to memref<*xf32>
    %diff = call @verifyMemRefF32(%C_cast, %C_ref_cast) : (memref<*xf32>, memref<*xf32>) -> i64
    call @printI64(%diff) : (i64) -> ()
    //call @printMemrefF32(%C_cast) : (memref<*xf32>) -> ()

    // CHECK: 0
    memref.dealloc %A_flatbytes : memref<2097152xi8>
    memref.dealloc %B : memref<2048x1024xi8>
    memref.dealloc %A_scale : memref<1024x128xf8E8M0FNU>
    memref.dealloc %B_scale : memref<128x1024xf8E8M0FNU>
    memref.dealloc %C : memref<1024x1024xf32>
    memref.dealloc %C_res : memref<1024x1024xf32>
    return
  }
  func.func private @verifyMemRefF32(%acutal : memref<*xf32>, %expected : memref<*xf32>) -> i64 attributes { llvm.emit_c_interface }
  func.func private @printI64(%num : i64) attributes { llvm.emit_c_interface }
  //func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes { llvm.emit_c_interface }

}
