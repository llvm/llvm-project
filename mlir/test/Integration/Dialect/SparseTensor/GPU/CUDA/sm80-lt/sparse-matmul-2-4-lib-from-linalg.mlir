//
// NOTE: this test requires gpu-sm80 and cusparselt
//
// DEFINE: %{compile} = mlir-opt %s \
// DEFINE: --sparse-compiler="enable-runtime-library=true enable-gpu-libgen gpu-triple=nvptx64-nvidia-cuda gpu-chip=sm_80 gpu-features=+ptx71 gpu-format=%gpu_compilation_format
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE: --shared-libs=%mlir_cuda_runtime \
// DEFINE: --shared-libs=%mlir_c_runner_utils \
// DEFINE: --e main --entry-point-result=void \
// DEFINE: | FileCheck %s

//  RUN:  %{compile}" | %{run}
//  RUN:  %{compile} gpu-data-transfer-strategy=pinned-dma" | %{run}
//  Tracker #64316
//  RUNNOT: %{compile} gpu-data-transfer-strategy=zero-copy" | %{run}

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

module {
  llvm.func @mgpuCreateSparseLtEnv()
  llvm.func @mgpuDestroySparseLtEnv()

  //
  // TODO: This uses our temporary ATTRIBUTE, replace with 2:4 type!
  //
  func.func @matmul_2to4(%arg0: tensor<16x32xf16>, %arg1: tensor<32x16xf16>, %arg2: tensor<16x16xf16>) -> tensor<16x16xf16> {
    %0 = linalg.generic { DENSE24, indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<16x32xf16>, tensor<32x16xf16>) outs(%arg2 : tensor<16x16xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %1 = arith.mulf %in, %in_0 : f16
      %2 = arith.addf %out, %1 : f16
      linalg.yield %2 : f16
    } -> tensor<16x16xf16>
    return %0 : tensor<16x16xf16>
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

    //
    // Setup matrix A.
    //
    %DA = tensor.generate {
    ^bb0(%i: index, %j: index):
      // (i+ j/2 + 1) if j %2 == 0 else 0
      %cf0 = arith.constant 0.0 : f16
      %cf1 = arith.constant 1.0 : f16
      %j_2 = arith.floordivsi %j, %c2 : index
      %quotient = arith.remsi %j, %c2 : index
      %sum = arith.addi %i, %j_2 : index
      %sum_i = arith.index_cast %sum : index to i64
      %sum_f = arith.uitofp %sum_i : i64 to f16
      %sum_f_plus1 = arith.addf %sum_f, %cf1 : f16
      %is_zero = arith.cmpi "eq", %quotient, %c0 : index
      %s = arith.select %is_zero, %sum_f_plus1, %cf0 : f16
      tensor.yield %s : f16
    } : tensor<16x32xf16>

    //
    // Setup matrix B.
    //
    %DB = tensor.generate {
    ^bb0(%i: index, %j: index):
      // if j_i >=8, j_i - 8 else 0
      %is_ge8 = arith.cmpi "sge", %j, %c8 : index
      %j_minus8 = arith.subi %j, %c8 : index
      %j2 = arith.select %is_ge8, %j_minus8, %j : index
      %r_i = arith.subi %j2, %i : index
      %r_i64 = arith.index_cast %r_i : index to i64
      %r_f = arith.sitofp %r_i64 : i64 to f16
      tensor.yield %r_f : f16
    } : tensor<32x16xf16>

    //
    // Reset matrix C.
    //
    %DC = tensor.generate {
    ^bb0(%i: index, %j: index):
      %cf0 = arith.constant 0.0 : f16
      tensor.yield %cf0 : f16
    } : tensor<16x16xf16>


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
      %pa0 = vector.transfer_read %DA[%pai, %c0], %f0 : tensor<16x32xf16>, vector<32xf16>
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
      %pb0 = vector.transfer_read %DB[%pbi, %c0], %f0 : tensor<32x16xf16>, vector<16xf16>
      vector.print %pb0 : vector<16xf16>
    }

    // Call the kernel.
    %t1  = arith.constant 1  : index
    %t32 = arith.constant 32 : index
    %c_out = call @matmul_2to4 (%DA, %DB, %DC): (tensor<16x32xf16>, tensor<32x16xf16>, tensor<16x16xf16>) -> tensor<16x16xf16>

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
      %pc0 = vector.transfer_read %c_out[%pci, %c0], %f0 : tensor<16x16xf16>, vector<16xf16>
      vector.print %pc0 : vector<16xf16>
    }
    
    llvm.call @mgpuDestroySparseLtEnv() : () -> ()
    return
  }
}
