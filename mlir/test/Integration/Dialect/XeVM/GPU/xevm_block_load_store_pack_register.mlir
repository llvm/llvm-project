// RUN: mlir-opt %s \
// RUN: | mlir-opt -pass-pipeline='builtin.module(cse,func.func(gpu-async-region),xevm-attach-target,gpu.module(convert-gpu-to-llvm-spv{use-64bit-index=true},convert-xevm-to-llvm,cse))' \
// RUN: | mlir-opt -convert-scf-to-cf -convert-cf-to-llvm -convert-vector-to-llvm -convert-arith-to-llvm \
// RUN: | mlir-opt -gpu-to-llvm -reconcile-unrealized-casts -cse -gpu-module-to-binary \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

module @gemm attributes {gpu.container_module} {
  gpu.module @kernel {
    gpu.func @block_load_store(%src: !llvm.ptr<1>, %dst: !llvm.ptr<1>) kernel  {
      %base_width = arith.constant 32 : i32 // bytewidth of the block
      %base_height_load = arith.constant 16 : i32 // number of rows
      %base_pitch = arith.constant 32 : i32 // bytewidth of the base row
      %x = arith.constant 0 : i32
      %y = arith.constant 0 : i32

      // Consider the following two loads:
      // Normal load:
      %loaded = xevm.blockload2d %src, %base_width, %base_height_load, %base_pitch, %x, %y
          <{elem_size_in_bits=16 : i32, tile_width=16 : i32, tile_height=16 : i32, v_blocks=1 : i32,
            transpose=false, pack_register=false}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
      %loaded_f16_flat = vector.bitcast %loaded : vector<16xi16> to vector<16xf16>
      %loaded_f16 = vector.shape_cast %loaded_f16_flat : vector<16xf16> to vector<8x1x2xf16>

      // Register packed load:
      %loaded_packed = xevm.blockload2d %src, %base_width, %base_height_load, %base_pitch, %x, %y
          <{elem_size_in_bits=16 : i32, tile_width=16 : i32, tile_height=16 : i32, v_blocks=1 : i32,
            transpose=false, pack_register=true}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
      %loaded_packed_f16_flat = vector.bitcast %loaded_packed : vector<8xi32> to vector<16xf16>
      %loaded_packed_f16 = vector.shape_cast %loaded_packed_f16_flat : vector<16xf16> to vector<8x1x2xf16>
      // Both can be represented the same way in code as vector<16xf16>.
      // A normal load pads a value to a dword (e.g., 32-bit) when loaded to a register.
      // Packed load "packs" multiple sub-dword values along the column (↓), allowing a single register
      // to hold multiple values.
      //  In SIMT, a work-item reads values along the column (↓), hence a sequence of values loaded by packing
      // to register is logically equivalent to the sequence of values loaded using a normal load.
      // The load results of both methods can have the same logical representation, but are expected to
      // differ in physical layout and register efficiency.

      %thread_x = gpu.thread_id x
      %thread_x_i64 = arith.index_cast %thread_x : index to i64
      %thread_x_i32 = llvm.trunc %thread_x_i64 : i64 to i32
      %thread_x_f16 = arith.sitofp %thread_x_i32 : i32 to f16
      %loaded_f16_modified = vector.insert %thread_x_f16, %loaded_packed_f16 [0,0,1] : f16 into vector<8x1x2xf16> // Both loaded_packed_f16 and loaded_f16 can be used here
      // We can only store [1,2,4,8]x[16] shapes for f16, so we have to do 2 stores
      %loaded_f16_modified_slice_0 = vector.extract_strided_slice %loaded_f16_modified
          {offsets = [0, 0, 0], sizes = [4, 1, 2], strides = [1, 1, 1]} : vector<8x1x2xf16> to vector<4x1x2xf16>
      %loaded_f16_modified_slice_0_flat = vector.shape_cast %loaded_f16_modified_slice_0 : vector<4x1x2xf16> to vector<8xf16>
      %base_height_store = arith.constant 8 : i32 // number of rows
      %base_width_store = arith.constant 32 : i32 // bytewidth of the block
      %base_pitch_store = arith.constant 32 : i32 // bytewidth of the base row
      xevm.blockstore2d %dst, %base_width_store, %base_height_store, %base_pitch_store, %x, %y, %loaded_f16_modified_slice_0_flat
          <{elem_size_in_bits=16 : i32, tile_width=16 : i32, tile_height=8 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xf16>)

      %loaded_f16_modified_slice_1 = vector.extract_strided_slice %loaded_f16_modified
          {offsets = [4, 0, 0], sizes = [4, 1, 2], strides = [1, 1, 1]} : vector<8x1x2xf16> to vector<4x1x2xf16>
      %loaded_f16_modified_slice_1_flat = vector.shape_cast %loaded_f16_modified_slice_1 : vector<4x1x2xf16> to vector<8xf16>

      %second_half_offset = arith.muli %base_pitch_store, %base_height_store : i32
      %second_half_ptr = llvm.getelementptr %dst[%second_half_offset] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i8
      xevm.blockstore2d %second_half_ptr, %base_width_store, %base_height_store, %base_pitch_store, %x, %y, %loaded_f16_modified_slice_1_flat
          <{elem_size_in_bits=16 : i32, tile_width=16 : i32, tile_height=8 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xf16>)
      gpu.return
    }
  }


  func.func @test(%src : memref<16x16xf16>) -> memref<16x16xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index // Multiple of the *maximum sub-group size* (see `intel_reqd_sub_group_size`)
    %memref_src = gpu.alloc() : memref<16x16xf16>
    gpu.memcpy %memref_src, %src : memref<16x16xf16>, memref<16x16xf16>
    %src_ptr_as_idx = memref.extract_aligned_pointer_as_index %memref_src : memref<16x16xf16> -> index
    %src_ptr_as_i64 = arith.index_cast %src_ptr_as_idx : index to i64
    %src_ptr = llvm.inttoptr %src_ptr_as_i64 : i64 to !llvm.ptr
    %src_ptr_casted = llvm.addrspacecast %src_ptr : !llvm.ptr to !llvm.ptr<1>

    %memref_dst = gpu.alloc() : memref<16x16xf16>
    %dst_ptr_as_idx = memref.extract_aligned_pointer_as_index %memref_dst : memref<16x16xf16> -> index
    %dst_ptr_as_i64 = arith.index_cast %dst_ptr_as_idx : index to i64
    %dst_ptr = llvm.inttoptr %dst_ptr_as_i64 : i64 to !llvm.ptr
    %dst_ptr_casted = llvm.addrspacecast %dst_ptr : !llvm.ptr to !llvm.ptr<1>

    gpu.launch_func @kernel::@block_load_store blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1)
        args(%src_ptr_casted : !llvm.ptr<1>, %dst_ptr_casted : !llvm.ptr<1>)
    gpu.dealloc %memref_src : memref<16x16xf16>
    %dst = memref.alloc() : memref<16x16xf16>
    gpu.memcpy %dst, %memref_dst : memref<16x16xf16>, memref<16x16xf16>
    gpu.dealloc %memref_dst : memref<16x16xf16>
    return %dst : memref<16x16xf16>
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %A = memref.alloc() : memref<16x16xf16>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 16 : index
    %c16 = arith.constant 16 : index
    %c11_f32 = arith.constant 11.1 : f16
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        memref.store %c11_f32, %A[%i, %j] : memref<16x16xf16>
      }
    }
    %B = call @test(%A) : (memref<16x16xf16>) -> memref<16x16xf16>
    %B_cast = memref.cast %B : memref<16x16xf16> to memref<*xf16>
    %A_cast = memref.cast %A : memref<16x16xf16> to memref<*xf16>
    call @printMemrefF16(%A_cast) : (memref<*xf16>) -> ()
    call @printMemrefF16(%B_cast) : (memref<*xf16>) -> ()

    // CHECK: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK-NEXT: [11.1{{.*}}]
    // CHECK-COUNT-224: 11.1
    // CHECK-NEXT: [11.1{{.*}}]

    // CHECK-NEXT: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK-NEXT: [11.1{{.*}}]
    // CHECK: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    // CHECK-COUNT-208: 11.1
    // CHECK-NEXT: [11.1{{.*}}]

    memref.dealloc %A : memref<16x16xf16>
    memref.dealloc %B : memref<16x16xf16>
    return
  }
  func.func private @printMemrefF16(%ptr : memref<*xf16>) attributes { llvm.emit_c_interface }
}
