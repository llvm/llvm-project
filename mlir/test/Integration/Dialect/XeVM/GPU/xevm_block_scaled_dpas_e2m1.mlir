// RUN: mlir-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=lane zebin-chip=cri" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// XFAIL: *
module @gemm attributes {gpu.container_module} {
  gpu.module @kernel {
    gpu.func @block_scaled_dpas_e2m1(%a: !llvm.ptr<1>, %b: !llvm.ptr<1>, %c: !llvm.ptr<1>) kernel {
      // TODO: some values are related can be derived from others like the following.
      // %M = arith.constant 8 : i32
      // %N = arith.constant 16 : i32
      // %K = arith.constant 8 : i32
      // %load_a_elem_bitwidth = arith.constant 32 : i32
      // %a_elem_bitwidth = arith.constant 16 : i32
      // %mx_elem_bitwidth = arith.constant 8 : i32
      // %load_a_pack_ratio = arith.divsi %load_a_elem_bitwidth, %a_elem_bitwidth : i32
      // %mx_pack_ratio = arith.divsi %load_a_elem_bitwidth, %mx_elem_bitwidth : i32
      // %load_a_K = arith.muli %K, %load_a_pack_ratio : i32
      // %load_b_K = arith.muli %K, %mx_pack_ratio : i32

      %x = arith.constant 0 : i32
      %y = arith.constant 0 : i32
      // Use constant A for simplifying the example. In real case, A will be loaded from memory.
      %loaded_a_casted = arith.constant dense<1.0> : vector<16xf16>
      %a_trunc_partial = xevm.truncf %loaded_a_casted { src_etype = f16, dst_etype = e2m1 } : (vector<16xf16>) -> vector<8xi8>
      %a_trunc = vector.shuffle %a_trunc_partial, %a_trunc_partial [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xi8>, vector<8xi8>
      // %a_trunc = arith.constant dense<0x22> : vector<16xi8>

      %a_trunc_casted = vector.bitcast %a_trunc : vector<16xi8> to vector<8xi16>

      %base_width_b = arith.constant 16 : i32
      %base_height_b = arith.constant 32 : i32
      %base_pitch_b = arith.constant 16 : i32
      // B is already in e2m1, and it will be used as is for MMA.
      // So the blockload2d op is configured to load normally with 8bit element bitwidth
      // with pack_register request.
      %loaded_b = xevm.blockload2d %b, %base_width_b, %base_height_b, %base_pitch_b, %x, %y
          <{elem_size_in_bits=8 : i32, tile_width=16 : i32, tile_height=32 : i32, v_blocks=1 : i32,
            transpose=false, pack_register=true}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>

      // Note: scale is not computed. Constant values are used for simplifying the example
      %scale_a = arith.constant dense<1.0> : vector<2xf8E8M0FNU>
      %scale_b = arith.constant dense<1.0> : vector<2xf8E8M0FNU>
      %scale_a_casted = vector.bitcast %scale_a : vector<2xf8E8M0FNU> to vector<2xi8>
      %scale_b_casted = vector.bitcast %scale_b : vector<2xf8E8M0FNU> to vector<2xi8>
      // Note: c is not loaded. constant vector is used for simplifying the example
      %loaded_c_casted = arith.constant dense<0.0> : vector<8xf32>

      %c_result = xevm.mma_mx %a_trunc_casted, %loaded_b, %scale_a_casted, %scale_b_casted, %loaded_c_casted
          {shape=<m=8, n=16, k=64>, types=<d=f32, a=e2m1, b=e2m1, c=f32>}
          : (vector<8xi16>, vector<8xi32>, vector<2xi8>, vector<2xi8>, vector<8xf32>) -> vector<8xf32>
      %c_result_casted = vector.bitcast %c_result : vector<8xf32> to vector<8xi32>

      %base_width_c = arith.constant 64 : i32
      %base_height_c = arith.constant 8 : i32
      %base_pitch_c = arith.constant 64 : i32
      xevm.blockstore2d %c, %base_width_c, %base_height_c, %base_pitch_c, %x, %y, %c_result_casted
          <{elem_size_in_bits=32 : i32, tile_width=16 : i32, tile_height=8 : i32}>
          : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
      gpu.return
    }
  }

  func.func @test(%a : memref<8x64xf16>, %b : memref<32x16xi8>, %c : memref<8x16xf32>) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index

    %memref_a = gpu.alloc() : memref<8x64xf16>
    gpu.memcpy %memref_a, %a : memref<8x64xf16>, memref<8x64xf16>
    %a_ptr_as_idx = memref.extract_aligned_pointer_as_index %memref_a : memref<8x64xf16> -> index
    %a_ptr_as_i64 = arith.index_cast %a_ptr_as_idx : index to i64
    %a_ptr = llvm.inttoptr %a_ptr_as_i64 : i64 to !llvm.ptr
    %a_ptr_casted = llvm.addrspacecast %a_ptr : !llvm.ptr to !llvm.ptr<1>

    %memref_b = gpu.alloc() : memref<32x16xi8>
    gpu.memcpy %memref_b, %b : memref<32x16xi8>, memref<32x16xi8>
    %b_ptr_as_idx = memref.extract_aligned_pointer_as_index %memref_b : memref<32x16xi8> -> index
    %b_ptr_as_i64 = arith.index_cast %b_ptr_as_idx : index to i64
    %b_ptr = llvm.inttoptr %b_ptr_as_i64 : i64 to !llvm.ptr
    %b_ptr_casted = llvm.addrspacecast %b_ptr : !llvm.ptr to !llvm.ptr<1>

    %memref_c = gpu.alloc() : memref<8x16xf32>
    gpu.memcpy %memref_c, %c : memref<8x16xf32>, memref<8x16xf32>
    %c_ptr_as_idx = memref.extract_aligned_pointer_as_index %memref_c : memref<8x16xf32> -> index
    %c_ptr_as_i64 = arith.index_cast %c_ptr_as_idx : index to i64
    %c_ptr = llvm.inttoptr %c_ptr_as_i64 : i64 to !llvm.ptr
    %c_ptr_casted = llvm.addrspacecast %c_ptr : !llvm.ptr to !llvm.ptr<1>

    gpu.launch_func @kernel::@block_scaled_dpas_e2m1 blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1)
        args(%a_ptr_casted : !llvm.ptr<1>, %b_ptr_casted : !llvm.ptr<1>, %c_ptr_casted : !llvm.ptr<1>)
    gpu.dealloc %memref_a : memref<8x64xf16>
    gpu.dealloc %memref_b : memref<32x16xi8>
    %res = memref.alloc() : memref<8x16xf32>
    gpu.memcpy %res, %memref_c : memref<8x16xf32>, memref<8x16xf32>
    gpu.dealloc %memref_c : memref<8x16xf32>
    return %res : memref<8x16xf32>
  }

  func.func @main() attributes {llvm.emit_c_interface} {

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c1f16 = arith.constant 1.0 : f16
    %c1e2m1packed = arith.constant 0x22 : i8
    %c0f32 = arith.constant 0.0 : f32

    %A = memref.alloc() : memref<8x64xf16>
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c64 step %c1 {
        memref.store %c1f16, %A[%i, %j] : memref<8x64xf16>
      }
    }

    %B = memref.alloc() : memref<32x16xi8>
    scf.for %i = %c0 to %c32 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        memref.store %c1e2m1packed, %B[%i, %j] : memref<32x16xi8>
      }
    }

    %C = memref.alloc() : memref<8x16xf32>
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        memref.store %c0f32, %C[%i, %j] : memref<8x16xf32>
      }
    }

    %C_res = call @test(%A, %B, %C) : (memref<8x64xf16>, memref<32x16xi8>, memref<8x16xf32>) -> memref<8x16xf32>
    %C_cast = memref.cast %C_res : memref<8x16xf32> to memref<*xf32>
    call @printMemrefF32(%C_cast) : (memref<*xf32>) -> ()

    // CHECK: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK-COUNT-8: [64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64]
    memref.dealloc %A : memref<8x64xf16>
    memref.dealloc %B : memref<32x16xi8>
    memref.dealloc %C : memref<8x16xf32>
    memref.dealloc %C_res : memref<8x16xf32>
    return
  }
  func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes { llvm.emit_c_interface }

}
