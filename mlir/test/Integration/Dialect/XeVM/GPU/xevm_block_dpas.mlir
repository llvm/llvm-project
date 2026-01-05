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
    // - Sets of `matrix_mad` intrinsics can differ based on device's *minimal* supported sub-group size.
    //   The *minimum supported* sub-group size should be used to call `matrix_mad` intrinsics.
    // https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_matrix_multiply_accumulate.html

    gpu.func @block_dpas(%a: !llvm.ptr<1>, %b: !llvm.ptr<1>, %c: !llvm.ptr<1>) kernel {
      %base_width_a = arith.constant 32 : i32
      %base_height_a = arith.constant 8 : i32
      %base_pitch_a = arith.constant 32 : i32
      %x = arith.constant 0 : i32
      %y = arith.constant 0 : i32
      %loaded_a = xevm.blockload2d %a, %base_width_a, %base_height_a, %base_pitch_a, %x, %y
          <{elem_size_in_bits=16 : i32, tile_width=16 : i32, tile_height=8 : i32, v_blocks=1 : i32,
            transpose=false, pack_register=false}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi16>

      %base_width_b = arith.constant 32 : i32
      %base_height_b = arith.constant 16 : i32
      %base_pitch_b = arith.constant 32 : i32
      %loaded_b1 = xevm.blockload2d %b, %base_width_b, %base_height_b, %base_pitch_b, %x, %y
          <{elem_size_in_bits=16 : i32, tile_width=16 : i32, tile_height=16 : i32, v_blocks=1 : i32,
            transpose=false, pack_register=false}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
      %loaded_b_casted = vector.bitcast %loaded_b1 : vector<16xi16> to vector<8xi32>

      %base_width_c = arith.constant 64 : i32
      %base_height_c = arith.constant 8 : i32
      %base_pitch_c = arith.constant 64 : i32
      %loaded_c = xevm.blockload2d %c, %base_width_c, %base_height_c, %base_pitch_c, %x, %y
          <{elem_size_in_bits=32 : i32, tile_width=16 : i32, tile_height=8 : i32, v_blocks=1 : i32,
            transpose=false, pack_register=false}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>

      %loaded_c_casted = vector.bitcast %loaded_c : vector<8xi32> to vector<8xf32>
      %c_result = xevm.mma %loaded_a, %loaded_b_casted, %loaded_c_casted
          {shape=<m=8, n=16, k=16>, types=<d=f32, a=f16, b=f16, c=f32>}
          : (vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32>
      %c_result_casted = vector.bitcast %c_result : vector<8xf32> to vector<8xi32>

      xevm.blockstore2d %c, %base_width_c, %base_height_c, %base_pitch_c, %x, %y, %c_result_casted
          <{elem_size_in_bits=32 : i32, tile_width=16 : i32, tile_height=8 : i32}>
          : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
      gpu.return
    }
  }

  func.func @test(%a : memref<8x16xf16>, %b : memref<16x16xf16>, %c : memref<8x16xf32>) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index

    %memref_a = gpu.alloc() : memref<8x16xf16>
    gpu.memcpy %memref_a, %a : memref<8x16xf16>, memref<8x16xf16>
    %a_ptr_as_idx = memref.extract_aligned_pointer_as_index %memref_a : memref<8x16xf16> -> index
    %a_ptr_as_i64 = arith.index_cast %a_ptr_as_idx : index to i64
    %a_ptr = llvm.inttoptr %a_ptr_as_i64 : i64 to !llvm.ptr
    %a_ptr_casted = llvm.addrspacecast %a_ptr : !llvm.ptr to !llvm.ptr<1>

    %memref_b = gpu.alloc() : memref<16x16xf16>
    gpu.memcpy %memref_b, %b : memref<16x16xf16>, memref<16x16xf16>
    %b_ptr_as_idx = memref.extract_aligned_pointer_as_index %memref_b : memref<16x16xf16> -> index
    %b_ptr_as_i64 = arith.index_cast %b_ptr_as_idx : index to i64
    %b_ptr = llvm.inttoptr %b_ptr_as_i64 : i64 to !llvm.ptr
    %b_ptr_casted = llvm.addrspacecast %b_ptr : !llvm.ptr to !llvm.ptr<1>

    %memref_c = gpu.alloc() : memref<8x16xf32>
    gpu.memcpy %memref_c, %c : memref<8x16xf32>, memref<8x16xf32>
    %c_ptr_as_idx = memref.extract_aligned_pointer_as_index %memref_c : memref<8x16xf32> -> index
    %c_ptr_as_i64 = arith.index_cast %c_ptr_as_idx : index to i64
    %c_ptr = llvm.inttoptr %c_ptr_as_i64 : i64 to !llvm.ptr
    %c_ptr_casted = llvm.addrspacecast %c_ptr : !llvm.ptr to !llvm.ptr<1>

    gpu.launch_func @kernel::@block_dpas blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1)
        args(%a_ptr_casted : !llvm.ptr<1>, %b_ptr_casted : !llvm.ptr<1>, %c_ptr_casted : !llvm.ptr<1>)
    gpu.dealloc %memref_a : memref<8x16xf16>
    gpu.dealloc %memref_b : memref<16x16xf16>
    %res = memref.alloc() : memref<8x16xf32>
    gpu.memcpy %res, %memref_c : memref<8x16xf32>, memref<8x16xf32>
    gpu.dealloc %memref_c : memref<8x16xf32>
    return %res : memref<8x16xf32>
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %A = memref.alloc() : memref<8x16xf16>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index

    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        %row_idx = arith.index_cast %i : index to i32
        %row = arith.sitofp %row_idx : i32 to f16
        memref.store %row, %A[%i, %j] : memref<8x16xf16>
      }
    }
    %B = memref.alloc() : memref<16x16xf16>
    scf.for %i = %c0 to %c16 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        %col_idx = arith.index_cast %j : index to i32
        %col = arith.sitofp %col_idx : i32 to f16
        memref.store %col, %B[%i, %j] : memref<16x16xf16>
      }
    }

    %C = memref.alloc() : memref<8x16xf32>
    %c0_f16 = arith.constant 0.0 : f32
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        memref.store %c0_f16, %C[%i, %j] : memref<8x16xf32>
      }
    }

    %C_res = call @test(%A, %B, %C) : (memref<8x16xf16>, memref<16x16xf16>, memref<8x16xf32>) -> memref<8x16xf32>
    %C_cast = memref.cast %C_res : memref<8x16xf32> to memref<*xf32>
    %A_cast = memref.cast %A : memref<8x16xf16> to memref<*xf16>
    call @printMemrefF32(%C_cast) : (memref<*xf32>) -> ()

    // CHECK: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK-NEXT: [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]
    // CHECK-NEXT: [0,   16,   32,   48,   64,   80,   96,   112,   128,   144,   160,   176,   192,   208,   224,   240]
    // CHECK-NEXT: [0,   32,   64,   96,   128,   160,   192,   224,   256,   288,   320,   352,   384,   416,   448,   480]
    // CHECK-NEXT: [0,   48,   96,   144,   192,   240,   288,   336,   384,   432,   480,   528,   576,   624,   672,   720]
    // CHECK-NEXT: [0,   64,   128,   192,   256,   320,   384,   448,   512,   576,   640,   704,   768,   832,   896,   960]
    // CHECK-NEXT: [0,   80,   160,   240,   320,   400,   480,   560,   640,   720,   800,   880,   960,   1040,   1120,   1200]
    // CHECK-NEXT: [0,   96,   192,   288,   384,   480,   576,   672,   768,   864,   960,   1056,   1152,   1248,   1344,   1440]
    // CHECK-NEXT: [0,   112,   224,   336,   448,   560,   672,   784,   896,   1008,   1120,   1232,   1344,   1456,   1568,   1680]

    memref.dealloc %A : memref<8x16xf16>
    memref.dealloc %B : memref<16x16xf16>
    memref.dealloc %C : memref<8x16xf32>
    memref.dealloc %C_res : memref<8x16xf32>
    return
  }
  func.func private @printMemrefF16(%ptr : memref<*xf16>) attributes { llvm.emit_c_interface }
  func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes { llvm.emit_c_interface }

}
