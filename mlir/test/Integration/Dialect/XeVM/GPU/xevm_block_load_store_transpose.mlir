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
      %base_height = arith.constant 16 : i32 // number of rows
      %base_pitch = arith.constant 32 : i32 // bytewidth of the base row
      %x = arith.constant 0 : i32
      %y = arith.constant 0 : i32
      // Normally a work-item loads a vertical slice (↓), but with *transpose* a work-item
      // loads a horizontal slice (→).
      // The tile dimension we want to slice must be a multiple of the sub-group size:
      //  e.g., we want to slice rows (→), then we need SG_SIZE % tile_height == 0.
      %loaded = xevm.blockload2d %src, %base_width, %base_height, %base_pitch, %x, %y
          <{elem_size_in_bits=32 : i32, tile_width=8 : i32, tile_height=16 : i32, v_blocks=1 : i32,
            transpose=true, pack_register=false}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
      %loaded_f32 = vector.bitcast %loaded : vector<8xi32> to vector<8xf32>

      %c0 = arith.constant 0 : i32
      %thread_x = gpu.thread_id x
      %thread_x_i64 = arith.index_cast %thread_x : index to i64
      %thread_x_i32 = llvm.trunc %thread_x_i64 : i64 to i32
      %thread_x_f32 = arith.sitofp %thread_x_i32 : i32 to f32
      %loaded_f32_modified = vector.insert %thread_x_f32, %loaded_f32[7] : f32 into vector<8xf32> // Use this to see where threadIds end up stored
      %loaded_f32_modified_1 = vector.bitcast %loaded_f32_modified : vector<8xf32> to vector<8xi32>

      %base_height_store = arith.constant 8 : i32 // number of rows
      %base_width_store = arith.constant 64 : i32 // bytewidth of the block
      %base_pitch_store = arith.constant 64 : i32 // bytewidth of the base row
      // "Transposed" stores are not available, meaning a work-item can store its vector as a vertical slice (↓).
      xevm.blockstore2d %dst, %base_width_store, %base_height_store, %base_pitch_store, %x, %y, %loaded
          <{elem_size_in_bits=32 : i32, tile_width=16 : i32, tile_height=8 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
      gpu.return
    }
  }


  func.func @test(%src : memref<16x8xf32>) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index // Multiple of the *maximum sub-group size* (see `intel_reqd_sub_group_size`)
    %memref_src = gpu.alloc() : memref<16x8xf32>
    gpu.memcpy %memref_src, %src : memref<16x8xf32>, memref<16x8xf32>
    %src_ptr_as_idx = memref.extract_aligned_pointer_as_index %memref_src : memref<16x8xf32> -> index
    %src_ptr_as_i64 = arith.index_cast %src_ptr_as_idx : index to i64
    %src_ptr = llvm.inttoptr %src_ptr_as_i64 : i64 to !llvm.ptr
    %src_ptr_casted = llvm.addrspacecast %src_ptr : !llvm.ptr to !llvm.ptr<1>

    %memref_dst = gpu.alloc() : memref<8x16xf32>
    %dst_ptr_as_idx = memref.extract_aligned_pointer_as_index %memref_dst : memref<8x16xf32> -> index
    %dst_ptr_as_i64 = arith.index_cast %dst_ptr_as_idx : index to i64
    %dst_ptr = llvm.inttoptr %dst_ptr_as_i64 : i64 to !llvm.ptr
    %dst_ptr_casted = llvm.addrspacecast %dst_ptr : !llvm.ptr to !llvm.ptr<1>

    gpu.launch_func @kernel::@block_load_store blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1)
        args(%src_ptr_casted : !llvm.ptr<1>, %dst_ptr_casted : !llvm.ptr<1>)
    gpu.dealloc %memref_src : memref<16x8xf32>
    %dst = memref.alloc() : memref<8x16xf32>
    gpu.memcpy %dst, %memref_dst : memref<8x16xf32>, memref<8x16xf32>
    gpu.dealloc %memref_dst : memref<8x16xf32>
    return %dst : memref<8x16xf32>
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %A = memref.alloc() : memref<16x8xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c11_f32 = arith.constant 11.11 : f16
    scf.for %i = %c0 to %c16 step %c1 {
      scf.for %j = %c0 to %c8 step %c1 {
        %c_10_f = arith.constant 10.0 : f32
        %j_i64 = arith.index_cast %j : index to i64
        %j_i32 = llvm.trunc %j_i64 : i64 to i32
        %j_f32 = arith.sitofp %j_i32 : i32 to f32
        %jj = arith.divf %j_f32, %c_10_f : f32

        %i_i64 = arith.index_cast %i : index to i64
        %i_i32 = llvm.trunc %i_i64 : i64 to i32
        %i_f32 = arith.sitofp %i_i32 : i32 to f32
        %ii = arith.addf %i_f32, %jj : f32
        memref.store %ii, %A[%i, %j] : memref<16x8xf32>
      }
    }
    %B = call @test(%A) : (memref<16x8xf32>) -> memref<8x16xf32>
    %A_cast = memref.cast %A : memref<16x8xf32> to memref<*xf32>
    %B_cast = memref.cast %B : memref<8x16xf32> to memref<*xf32>
    call @printMemrefF32(%A_cast) : (memref<*xf32>) -> ()
    // CHECK: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK-NEXT: [0,   0.1,   0.2,   0.3,   0.4,   0.5,   0.6,   0.7],
    // CHECK-NEXT: [1,   1.1,   1.2,   1.3,   1.4,   1.5,   1.6,   1.7],
    // CHECK-NEXT: [2,   2.1,   2.2,   2.3,   2.4,   2.5,   2.6,   2.7],
    // CHECK-NEXT: [3,   3.1,   3.2,   3.3,   3.4,   3.5,   3.6,   3.7],
    // CHECK-NEXT: [4,   4.1,   4.2,   4.3,   4.4,   4.5,   4.6,   4.7],
    // CHECK-NEXT: [5,   5.1,   5.2,   5.3,   5.4,   5.5,   5.6,   5.7],
    // CHECK-NEXT: [6,   6.1,   6.2,   6.3,   6.4,   6.5,   6.6,   6.7],
    // CHECK-NEXT: [7,   7.1,   7.2,   7.3,   7.4,   7.5,   7.6,   7.7],
    // CHECK-NEXT: [8,   8.1,   8.2,   8.3,   8.4,   8.5,   8.6,   8.7],
    // CHECK-NEXT: [9,   9.1,   9.2,   9.3,   9.4,   9.5,   9.6,   9.7],
    // CHECK-NEXT: [10,   10.1,   10.2,   10.3,   10.4,   10.5,   10.6,   10.7],
    // CHECK-NEXT: [11,   11.1,   11.2,   11.3,   11.4,   11.5,   11.6,   11.7],
    // CHECK-NEXT: [12,   12.1,   12.2,   12.3,   12.4,   12.5,   12.6,   12.7],
    // CHECK-NEXT: [13,   13.1,   13.2,   13.3,   13.4,   13.5,   13.6,   13.7],
    // CHECK-NEXT: [14,   14.1,   14.2,   14.3,   14.4,   14.5,   14.6,   14.7],
    // CHECK-NEXT: [15,   15.1,   15.2,   15.3,   15.4,   15.5,   15.6,   15.7]

    call @printMemrefF32(%B_cast) : (memref<*xf32>) -> ()
    // CHECK: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK-NEXT: [0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,   11,   12,   13,   14,   15],
    // CHECK-NEXT: [0.1,   1.1,   2.1,   3.1,   4.1,   5.1,   6.1,   7.1,   8.1,   9.1,   10.1,   11.1,   12.1,   13.1,   14.1,   15.1],
    // CHECK-NEXT: [0.2,   1.2,   2.2,   3.2,   4.2,   5.2,   6.2,   7.2,   8.2,   9.2,   10.2,   11.2,   12.2,   13.2,   14.2,   15.2],
    // CHECK-NEXT: [0.3,   1.3,   2.3,   3.3,   4.3,   5.3,   6.3,   7.3,   8.3,   9.3,   10.3,   11.3,   12.3,   13.3,   14.3,   15.3],
    // CHECK-NEXT: [0.4,   1.4,   2.4,   3.4,   4.4,   5.4,   6.4,   7.4,   8.4,   9.4,   10.4,   11.4,   12.4,   13.4,   14.4,   15.4],
    // CHECK-NEXT: [0.5,   1.5,   2.5,   3.5,   4.5,   5.5,   6.5,   7.5,   8.5,   9.5,   10.5,   11.5,   12.5,   13.5,   14.5,   15.5],
    // CHECK-NEXT: [0.6,   1.6,   2.6,   3.6,   4.6,   5.6,   6.6,   7.6,   8.6,   9.6,   10.6,   11.6,   12.6,   13.6,   14.6,   15.6],
    // CHECK-NEXT: [0.7,   1.7,   2.7,   3.7,   4.7,   5.7,   6.7,   7.7,   8.7,   9.7,   10.7,   11.7,   12.7,   13.7,   14.7,   15.7]

    memref.dealloc %A : memref<16x8xf32>
    memref.dealloc %B : memref<8x16xf32>
    return
  }
  func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes { llvm.emit_c_interface }
}
