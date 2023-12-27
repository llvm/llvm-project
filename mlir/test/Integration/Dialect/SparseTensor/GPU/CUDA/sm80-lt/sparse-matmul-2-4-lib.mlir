// NOTE: this test requires gpu-sm80 and cusparselt
//
// DEFINE: %{compile} = mlir-opt --convert-vector-to-scf --convert-scf-to-cf -convert-cf-to-llvm --convert-vector-to-llvm \
// DEFINE: --convert-arith-to-llvm --gpu-to-llvm --reconcile-unrealized-casts \
// DEFINE: %s
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:   --shared-libs=%mlir_cuda_runtime \
// DEFINE:   --shared-libs=%mlir_c_runner_utils \
// DEFINE:   --e main --entry-point-result=void \
// DEFINE: | FileCheck %s
//
// RUN: %{compile} | %{run}

module {
  llvm.func @mgpuCreateSparseLtEnv()
  llvm.func @mgpuDestroySparseLtEnv()

  func.func @sampled_matmul(%a : memref<16x32xf16>,
                            %b : memref<32x16xf16>,
                            %c : memref<16x16xf16>) {
    %c0  = arith.constant 0.0 : f16
    %c1  = arith.constant 1   : index
    %c2  = arith.constant 2   : index
    %c8  = arith.constant 8   : index
    %c16 = arith.constant 16  : index
    %c32 = arith.constant 32  : index
    %c1048576 = arith.constant 1048576 : index
    %token0 = gpu.wait async
    %d_a, %token1 = gpu.alloc async [%token0] () : memref<16x32xf16>
    %d_b, %token2 = gpu.alloc async [%token1] () : memref<32x16xf16>
    %d_c, %token3 = gpu.alloc async [%token2] () : memref<16x16xf16>
    %token4 = gpu.memcpy async [%token3] %d_a, %a : memref<16x32xf16>, memref<16x32xf16>
    %token5 = gpu.memcpy async [%token4] %d_b, %b : memref<32x16xf16>, memref<32x16xf16>
    %token6 = gpu.memcpy async [%token5] %d_c, %c : memref<16x16xf16>, memref<16x16xf16>
    %spmat, %token8 = gpu.create_2to4_spmat async  [%token6]{PRUNE_AND_CHECK} %c16, %c32, %d_a: memref<16x32xf16>
    %dnmat, %token9 = gpu.create_dn_tensor async [%token8] %d_b, %c32, %c16: index, index into memref<32x16xf16>
    %dnmat2, %token10 = gpu.create_dn_tensor async [%token9] %d_c, %c16, %c16: index, index into memref<16x16xf16>
    %bufferSz0, %bufferSz1, %bufferSz2, %token11 = gpu.spmm_buffer_size async [%token10] %spmat{NON_TRANSPOSE}, %dnmat{NON_TRANSPOSE}, %dnmat2 : index, index,index into f16
    %mem1, %token12 = gpu.alloc async [%token11] (%bufferSz0) : memref<?xf16>
    %mem2, %token13 = gpu.alloc async [%token12] (%bufferSz1) : memref<?xf16>
    %mem3, %token14 = gpu.alloc async [%token13] (%bufferSz2) : memref<?xf16>
    %token15 = gpu.spmm async [%token14] %spmat{NON_TRANSPOSE}, %dnmat{NON_TRANSPOSE}, %dnmat2, %mem1, %mem2, %mem3 : memref<?xf16>, memref<?xf16>,memref<?xf16> into f16
    %token16 = gpu.destroy_sp_mat async [%token15] %spmat
    %token17 = gpu.destroy_dn_tensor async [%token16] %dnmat
    %token18 = gpu.destroy_dn_tensor async [%token17] %dnmat2
    %token19 = gpu.memcpy async [%token18] %c, %d_c : memref<16x16xf16>, memref<16x16xf16>
    %token20 = gpu.dealloc async [%token19] %d_c : memref<16x16xf16>
    %token21 = gpu.dealloc async [%token20] %d_b : memref<32x16xf16>
    %token22 = gpu.dealloc async [%token21] %d_a : memref<16x32xf16>
    %token23 = gpu.dealloc async [%token22] %mem3 : memref<?xf16>
    %token24 = gpu.dealloc async [%token23] %mem2 : memref<?xf16>
    %token25 = gpu.dealloc async [%token24] %mem1 : memref<?xf16>
    gpu.wait [%token25]
    return
  }

  //
  // This test performs a matrix multiplication
  //   C = A x B
  // using NVidia 2:4 structured sparsity for A.
  //
  func.func @main() {
    llvm.call @mgpuCreateSparseLtEnv() : () -> ()
    %f0  = arith.constant 0.0 : f16
    %c0  = arith.constant 0   : index
    %c1  = arith.constant 1   : index
    %c2  = arith.constant 2   : index
    %c8  = arith.constant 8   : index
    %c16 = arith.constant 16  : index
    %c32 = arith.constant 32  : index
    %c64 = arith.constant 64  : index

    // Matrices A, B, C (16x32, 32x16, 16x16).
    %a = memref.alloc() : memref<16x32xf16> // 16x32 with 2:4, row-major
    %b = memref.alloc() : memref<32x16xf16> // regular dense   column-major
    %c = memref.alloc() : memref<16x16xf16> // accumulator     row-major

    //
    // Setup matrix A.
    //
    scf.for %ai = %c0 to %c16 step %c1 {
      scf.for %aj = %c0 to %c16 step %c1 {
        %cf0  = arith.constant 0.0: f16
        %a0 = arith.addi %ai, %aj : index
        %a1 = arith.addi %a0, %c1 : index
        %a2 = arith.index_cast %a1 : index to i32
        %a3 = arith.sitofp %a2 : i32 to f16
        %ajj = arith.muli %aj, %c2 : index
        %ajj2 = arith.addi %ajj, %c1 : index
        memref.store %a3, %a[%ai, %ajj] : memref<16x32xf16>
        memref.store %cf0, %a[%ai, %ajj2] : memref<16x32xf16>
      }
    }

    //
    // Setup matrix B.
    //
    scf.for %bi = %c0 to %c8 step %c1 {
      scf.for %bj = %c0 to %c32 step %c1 {
        %b0 = arith.subi %bi, %bj : index
        %b1 = arith.index_cast %b0 : index to i32
        %b2 = arith.sitofp %b1 : i32 to f16
        %bii = arith.addi %bi, %c8 : index
        memref.store %b2, %b[%bj, %bi] : memref<32x16xf16>
        memref.store %b2, %b[%bj, %bii] : memref<32x16xf16>
      }
    }

    //
    // Reset matrix C.
    //
    scf.for %ci = %c0 to %c16 step %c1 {
      scf.for %cj = %c0 to %c16 step %c1 {
        memref.store %f0, %c[%ci, %cj] : memref<16x16xf16>
      }
    }

    //
    // Sanity check on 16x32 full 2:4 input matrix A.
    //
    //
    // CHECK:      ( 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16, 0 )
    // CHECK-NEXT: ( 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16, 0, 17, 0 )
    // CHECK-NEXT: ( 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16, 0, 17, 0, 18, 0 )
    // CHECK-NEXT: ( 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16, 0, 17, 0, 18, 0, 19, 0 )
    // CHECK-NEXT: ( 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16, 0, 17, 0, 18, 0, 19, 0, 20, 0 )
    // CHECK-NEXT: ( 6, 0, 7, 0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16, 0, 17, 0, 18, 0, 19, 0, 20, 0, 21, 0 )
    // CHECK-NEXT: ( 7, 0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16, 0, 17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0 )
    // CHECK-NEXT: ( 8, 0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16, 0, 17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23, 0 )
    // CHECK-NEXT: ( 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16, 0, 17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23, 0, 24, 0 )
    // CHECK-NEXT: ( 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16, 0, 17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23, 0, 24, 0, 25, 0 )
    // CHECK-NEXT: ( 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16, 0, 17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23, 0, 24, 0, 25, 0, 26, 0 )
    // CHECK-NEXT: ( 12, 0, 13, 0, 14, 0, 15, 0, 16, 0, 17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23, 0, 24, 0, 25, 0, 26, 0, 27, 0 )
    // CHECK-NEXT: ( 13, 0, 14, 0, 15, 0, 16, 0, 17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23, 0, 24, 0, 25, 0, 26, 0, 27, 0, 28, 0 )
    // CHECK-NEXT: ( 14, 0, 15, 0, 16, 0, 17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23, 0, 24, 0, 25, 0, 26, 0, 27, 0, 28, 0, 29, 0 )
    // CHECK-NEXT: ( 15, 0, 16, 0, 17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23, 0, 24, 0, 25, 0, 26, 0, 27, 0, 28, 0, 29, 0, 30, 0 )
    // CHECK-NEXT: ( 16, 0, 17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23, 0, 24, 0, 25, 0, 26, 0, 27, 0, 28, 0, 29, 0, 30, 0, 31, 0 )
    //
    scf.for %pai = %c0 to %c16 step %c1 {
      %pa0 = vector.transfer_read %a[%pai, %c0], %f0 : memref<16x32xf16>, vector<32xf16>
      vector.print %pa0 : vector<32xf16>
    }

    //
    // Sanity check on input matrix 32x16 B.
    //
    // CHECK-NEXT: (   0,   1,   2,   3,   4,   5,   6,   7,   0,   1,   2,   3,   4,   5,   6,   7 )
    // CHECK-NEXT: (  -1,   0,   1,   2,   3,   4,   5,   6,  -1,   0,   1,   2,   3,   4,   5,   6 )
    // CHECK-NEXT: (  -2,  -1,   0,   1,   2,   3,   4,   5,  -2,  -1,   0,   1,   2,   3,   4,   5 )
    // CHECK-NEXT: (  -3,  -2,  -1,   0,   1,   2,   3,   4,  -3,  -2,  -1,   0,   1,   2,   3,   4 )
    // CHECK-NEXT: (  -4,  -3,  -2,  -1,   0,   1,   2,   3,  -4,  -3,  -2,  -1,   0,   1,   2,   3 )
    // CHECK-NEXT: (  -5,  -4,  -3,  -2,  -1,   0,   1,   2,  -5,  -4,  -3,  -2,  -1,   0,   1,   2 )
    // CHECK-NEXT: (  -6,  -5,  -4,  -3,  -2,  -1,   0,   1,  -6,  -5,  -4,  -3,  -2,  -1,   0,   1 )
    // CHECK-NEXT: (  -7,  -6,  -5,  -4,  -3,  -2,  -1,   0,  -7,  -6,  -5,  -4,  -3,  -2,  -1,   0 )
    // CHECK-NEXT: (  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1 )
    // CHECK-NEXT: (  -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2 )
    // CHECK-NEXT: ( -10,  -9,  -8,  -7,  -6,  -5,  -4,  -3, -10,  -9,  -8,  -7,  -6,  -5,  -4,  -3 )
    // CHECK-NEXT: ( -11, -10,  -9,  -8,  -7,  -6,  -5,  -4, -11, -10,  -9,  -8,  -7,  -6,  -5,  -4 )
    // CHECK-NEXT: ( -12, -11, -10,  -9,  -8,  -7,  -6,  -5, -12, -11, -10,  -9,  -8,  -7,  -6,  -5 )
    // CHECK-NEXT: ( -13, -12, -11, -10,  -9,  -8,  -7,  -6, -13, -12, -11, -10,  -9,  -8,  -7,  -6 )
    // CHECK-NEXT: ( -14, -13, -12, -11, -10,  -9,  -8,  -7, -14, -13, -12, -11, -10,  -9,  -8,  -7 )
    // CHECK-NEXT: ( -15, -14, -13, -12, -11, -10,  -9,  -8, -15, -14, -13, -12, -11, -10,  -9,  -8 )
    // CHECK-NEXT: ( -16, -15, -14, -13, -12, -11, -10,  -9, -16, -15, -14, -13, -12, -11, -10,  -9 )
    // CHECK-NEXT: ( -17, -16, -15, -14, -13, -12, -11, -10, -17, -16, -15, -14, -13, -12, -11, -10 )
    // CHECK-NEXT: ( -18, -17, -16, -15, -14, -13, -12, -11, -18, -17, -16, -15, -14, -13, -12, -11 )
    // CHECK-NEXT: ( -19, -18, -17, -16, -15, -14, -13, -12, -19, -18, -17, -16, -15, -14, -13, -12 )
    // CHECK-NEXT: ( -20, -19, -18, -17, -16, -15, -14, -13, -20, -19, -18, -17, -16, -15, -14, -13 )
    // CHECK-NEXT: ( -21, -20, -19, -18, -17, -16, -15, -14, -21, -20, -19, -18, -17, -16, -15, -14 )
    // CHECK-NEXT: ( -22, -21, -20, -19, -18, -17, -16, -15, -22, -21, -20, -19, -18, -17, -16, -15 )
    // CHECK-NEXT: ( -23, -22, -21, -20, -19, -18, -17, -16, -23, -22, -21, -20, -19, -18, -17, -16 )
    // CHECK-NEXT: ( -24, -23, -22, -21, -20, -19, -18, -17, -24, -23, -22, -21, -20, -19, -18, -17 )
    // CHECK-NEXT: ( -25, -24, -23, -22, -21, -20, -19, -18, -25, -24, -23, -22, -21, -20, -19, -18 )
    // CHECK-NEXT: ( -26, -25, -24, -23, -22, -21, -20, -19, -26, -25, -24, -23, -22, -21, -20, -19 )
    // CHECK-NEXT: ( -27, -26, -25, -24, -23, -22, -21, -20, -27, -26, -25, -24, -23, -22, -21, -20 )
    // CHECK-NEXT: ( -28, -27, -26, -25, -24, -23, -22, -21, -28, -27, -26, -25, -24, -23, -22, -21 )
    // CHECK-NEXT: ( -29, -28, -27, -26, -25, -24, -23, -22, -29, -28, -27, -26, -25, -24, -23, -22 )
    // CHECK-NEXT: ( -30, -29, -28, -27, -26, -25, -24, -23, -30, -29, -28, -27, -26, -25, -24, -23 )
    // CHECK-NEXT: ( -31, -30, -29, -28, -27, -26, -25, -24, -31, -30, -29, -28, -27, -26, -25, -24 )
    //
    //
    scf.for %pbi = %c0 to %c32 step %c1 {
      %pb0 = vector.transfer_read %b[%pbi, %c0], %f0 : memref<32x16xf16>, vector<16xf16>
      vector.print %pb0 : vector<16xf16>
    }

    // Call the kernel.
    call @sampled_matmul (%a, %b, %c): (memref<16x32xf16>, memref<32x16xf16>, memref<16x16xf16>) -> ()

    //
    // Verify computed matrix C.
    //
    // CHECK-NEXT: ( -2720, -2584, -2448, -2312, -2176, -2040, -1904, -1768, -2720, -2584, -2448, -2312, -2176, -2040, -1904, -1768  )
    // CHECK-NEXT: ( -2960, -2808, -2656, -2504, -2352, -2200, -2048, -1896, -2960, -2808, -2656, -2504, -2352, -2200, -2048, -1896  )
    // CHECK-NEXT: ( -3200, -3032, -2864, -2696, -2528, -2360, -2192, -2024, -3200, -3032, -2864, -2696, -2528, -2360, -2192, -2024  )
    // CHECK-NEXT: ( -3440, -3256, -3072, -2888, -2704, -2520, -2336, -2152, -3440, -3256, -3072, -2888, -2704, -2520, -2336, -2152  )
    // CHECK-NEXT: ( -3680, -3480, -3280, -3080, -2880, -2680, -2480, -2280, -3680, -3480, -3280, -3080, -2880, -2680, -2480, -2280  )
    // CHECK-NEXT: ( -3920, -3704, -3488, -3272, -3056, -2840, -2624, -2408, -3920, -3704, -3488, -3272, -3056, -2840, -2624, -2408  )
    // CHECK-NEXT: ( -4160, -3928, -3696, -3464, -3232, -3000, -2768, -2536, -4160, -3928, -3696, -3464, -3232, -3000, -2768, -2536  )
    // CHECK-NEXT: ( -4400, -4152, -3904, -3656, -3408, -3160, -2912, -2664, -4400, -4152, -3904, -3656, -3408, -3160, -2912, -2664  )
    // CHECK-NEXT: ( -4640, -4376, -4112, -3848, -3584, -3320, -3056, -2792, -4640, -4376, -4112, -3848, -3584, -3320, -3056, -2792  )
    // CHECK-NEXT: ( -4880, -4600, -4320, -4040, -3760, -3480, -3200, -2920, -4880, -4600, -4320, -4040, -3760, -3480, -3200, -2920  )
    // CHECK-NEXT: ( -5120, -4824, -4528, -4232, -3936, -3640, -3344, -3048, -5120, -4824, -4528, -4232, -3936, -3640, -3344, -3048  )
    // CHECK-NEXT: ( -5360, -5048, -4736, -4424, -4112, -3800, -3488, -3176, -5360, -5048, -4736, -4424, -4112, -3800, -3488, -3176  )
    // CHECK-NEXT: ( -5600, -5272, -4944, -4616, -4288, -3960, -3632, -3304, -5600, -5272, -4944, -4616, -4288, -3960, -3632, -3304  )
    // CHECK-NEXT: ( -5840, -5496, -5152, -4808, -4464, -4120, -3776, -3432, -5840, -5496, -5152, -4808, -4464, -4120, -3776, -3432  )
    // CHECK-NEXT: ( -6080, -5720, -5360, -5000, -4640, -4280, -3920, -3560, -6080, -5720, -5360, -5000, -4640, -4280, -3920, -3560  )
    // CHECK-NEXT: ( -6320, -5944, -5568, -5192, -4816, -4440, -4064, -3688, -6320, -5944, -5568, -5192, -4816, -4440, -4064, -3688  )
    //
    scf.for %pci = %c0 to %c16 step %c1 {
      %pc0 = vector.transfer_read %c[%pci, %c0], %f0 : memref<16x16xf16>, vector<16xf16>
      vector.print %pc0 : vector<16xf16>
    }

    llvm.call @mgpuDestroySparseLtEnv() : () -> ()
    return
  }
}
