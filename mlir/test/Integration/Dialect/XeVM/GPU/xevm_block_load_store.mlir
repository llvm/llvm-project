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
    // - `cl_intel_subgroups` block load/store intrinsics operate at the *maximum* sub-group size,
    //     regardless of the active sub-group size. Make sure `clGetKernelSubGroupInfo` meets your expectations.
    // - The attribute `intel_reqd_sub_group_size` establishes the maximum sub-group size for a kernel.
    //
    // Note: launching 16 threads without explicit `intel_reqd_sub_group_size = 16` may still use
    //       the default sub-group size of 32.
    //
    // https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_required_subgroup_size.html
    // https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroups.html

    gpu.func @block_load_store(%src: !llvm.ptr<1>, %dst: !llvm.ptr<1>) kernel  {
      %base_width = arith.constant 64 : i32 // bytewidth of the block
      %base_height = arith.constant 8 : i32 // number of rows
      %base_pitch = arith.constant 64 : i32 // bytewidth of the base row
      %x = arith.constant 0 : i32
      %y = arith.constant 0 : i32
      // If `intel_reqd_sub_group_size = 16` is not set, the default (32) is used and this `blockload2d`
      // would only load 4 elements into vector<8xi32>
      %loaded = xevm.blockload2d %src, %base_width, %base_height, %base_pitch, %x, %y
          <{elem_size_in_bits=32 : i32, tile_width=16 : i32, tile_height=8 : i32, v_blocks=1 : i32,
            transpose=false, pack_register=false}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
      %loaded_f32 = vector.bitcast %loaded : vector<8xi32> to vector<8xf32>
      %c0 = arith.constant 0 : index
      %thread_x = gpu.thread_id x
      %thread_x_i64 = arith.index_cast %thread_x : index to i64
      %thread_x_i32 = llvm.trunc %thread_x_i64 : i64 to i32
      %thread_x_f32 = arith.sitofp %thread_x_i32 : i32 to f32
      %loaded_f32_modified = vector.insert %thread_x_f32, %loaded_f32[%c0] : f32 into vector<8xf32>
      %loaded_modified = vector.bitcast %loaded_f32_modified : vector<8xf32> to vector<8xi32>
      xevm.blockstore2d %dst, %base_width, %base_height, %base_pitch, %x, %y, %loaded_modified
          <{elem_size_in_bits=32 : i32, tile_width=16 : i32, tile_height=8 : i32}>
          : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
      gpu.return
    }
  }

  func.func @test(%src : memref<8x16xf32>) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index // Multiple of the *maximum sub-group size* (see `intel_reqd_sub_group_size`)
    %memref_src = gpu.alloc() : memref<8x16xf32>
    gpu.memcpy %memref_src, %src : memref<8x16xf32>, memref<8x16xf32>
    %src_ptr_as_idx = memref.extract_aligned_pointer_as_index %memref_src : memref<8x16xf32> -> index
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
    gpu.dealloc %memref_src : memref<8x16xf32>
    %dst = memref.alloc() : memref<8x16xf32>
    gpu.memcpy %dst, %memref_dst : memref<8x16xf32>, memref<8x16xf32>
    gpu.dealloc %memref_dst : memref<8x16xf32>
    return %dst : memref<8x16xf32>
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %A = memref.alloc() : memref<8x16xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c11_f32 = arith.constant 11.11 : f32
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        memref.store %c11_f32, %A[%i, %j] : memref<8x16xf32>
      }
    }
    %B = call @test(%A) : (memref<8x16xf32>) -> memref<8x16xf32>
    %B_cast = memref.cast %B : memref<8x16xf32> to memref<*xf32>
    %A_cast = memref.cast %A : memref<8x16xf32> to memref<*xf32>
    call @printMemrefF32(%A_cast) : (memref<*xf32>) -> ()
    call @printMemrefF32(%B_cast) : (memref<*xf32>) -> ()

    // CHECK: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK-NEXT: [11.11{{.*}}]
    // CHECK-COUNT-96: 11.11
    // CHECK-NEXT: [11.11{{.*}}]

    // CHECK-NEXT: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    // CHECK-COUNT-96: 11.11
    // CHECK-NEXT: [11.11{{.*}}]

    memref.dealloc %A : memref<8x16xf32>
    memref.dealloc %B : memref<8x16xf32>
    return
  }
  func.func private @printMemrefF32(%ptr : memref<*xf32>)
}
