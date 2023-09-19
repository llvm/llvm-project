// RUN: mlir-opt %s \
// RUN: | mlir-opt -convert-scf-to-cf \
// RUN: | mlir-opt -gpu-kernel-outlining \
// RUN: | mlir-opt -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-amdgpu{chipset=%chip index-bitwidth=32},convert-gpu-to-rocdl{chipset=%chip index-bitwidth=32},reconcile-unrealized-casts),rocdl-attach-target{chip=%chip})' \
// RUN: | mlir-opt -gpu-to-llvm -gpu-module-to-binary \
// RUN: | mlir-cpu-runner \
// RUN:   --shared-libs=%mlir_rocm_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

func.func @main() {
  %0 = memref.alloc() : memref<16x16xf16>
  %22 = memref.alloc() : memref<16x16xf32>

  %f1 = arith.constant 1.0e+00 : f16
  %f0 = arith.constant 0.0e+00 : f32
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index

  // Intialize the Input matrix with ones.
  scf.for %arg0 = %c0 to %c16 step %c1 {
    scf.for %arg1 = %c0 to %c16 step %c1 {
      %cast_c = arith.index_cast %arg1 : index to i16
      %cast_r = arith.index_cast %arg0 : index to i16
      %add = arith.addi %cast_r, %cast_c : i16
      %float = arith.sitofp %add : i16 to f16
      memref.store %float, %0[%arg0, %arg1] : memref<16x16xf16>
    }
  }
  // Intialize the accumulator matrix with zeros.
  scf.for %arg0 = %c0 to %c16 step %c1 {
    scf.for %arg1 = %c0 to %c16 step %c1 {
      memref.store %f0, %22[%arg0, %arg1] : memref<16x16xf32>
    }
  }

  %2 = memref.cast %0 : memref<16x16xf16> to memref<*xf16>
  %33 = memref.cast %22 : memref<16x16xf32> to memref<*xf32>

  %stream = gpu.wait async
  %gpu_in, %asyncToken_0 = gpu.alloc async [%stream] () : memref<16x16xf16>
  %gpu_out, %asyncToken_1 = gpu.alloc async [%stream] () : memref<16x16xf32>

  %asyncToken_2 = gpu.memcpy async [%stream] %gpu_in, %0 : memref<16x16xf16>, memref<16x16xf16>
  %asyncToken_3 = gpu.memcpy async [%stream] %gpu_out, %22 : memref<16x16xf32>, memref<16x16xf32>

  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c32, %block_y = %c1, %block_z = %c1) {
    %A = gpu.subgroup_mma_load_matrix %gpu_in[%c0, %c0] {leadDimension = 16 : index} : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "AOp">
    %B = gpu.subgroup_mma_load_matrix %gpu_in[%c0, %c0] {leadDimension = 16 : index} : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "BOp">
    %C = gpu.subgroup_mma_load_matrix %gpu_out[%c0, %c0] {leadDimension = 16 : index} : memref<16x16xf32> -> !gpu.mma_matrix<16x16xf32, "COp">

    %R = gpu.subgroup_mma_compute %A, %B, %C : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">

    gpu.subgroup_mma_store_matrix %R, %gpu_out[%c0, %c0] {leadDimension = 16 : index}: !gpu.mma_matrix<16x16xf32, "COp">, memref<16x16xf32>
    gpu.terminator
  }

  %asyncToken_4 = gpu.memcpy async [%stream] %22, %gpu_out : memref<16x16xf32>, memref<16x16xf32>
  gpu.wait [%stream]

  // Print the memref after computation.
  call @printMemrefF32(%33) : (memref<*xf32>) -> ()
  // CHECK:        [1240,   1360,   1480,   1600,   1720,   1840,   1960,   2080,   2200,   2320,   2440,   2560,   2680,   2800,   2920,   3040],
  // CHECK-NEXT:   [1360,   1496,   1632,   1768,   1904,   2040,   2176,   2312,   2448,   2584,   2720,   2856,   2992,   3128,   3264,   3400],
  // CHECK-NEXT:   [1480,   1632,   1784,   1936,   2088,   2240,   2392,   2544,   2696,   2848,   3000,   3152,   3304,   3456,   3608,   3760],
  // CHECK-NEXT:   [1600,   1768,   1936,   2104,   2272,   2440,   2608,   2776,   2944,   3112,   3280,   3448,   3616,   3784,   3952,   4120],
  // CHECK-NEXT:   [1720,   1904,   2088,   2272,   2456,   2640,   2824,   3008,   3192,   3376,   3560,   3744,   3928,   4112,   4296,   4480],
  // CHECK-NEXT:   [1840,   2040,   2240,   2440,   2640,   2840,   3040,   3240,   3440,   3640,   3840,   4040,   4240,   4440,   4640,   4840],
  // CHECK-NEXT:   [1960,   2176,   2392,   2608,   2824,   3040,   3256,   3472,   3688,   3904,   4120,   4336,   4552,   4768,   4984,   5200],
  // CHECK-NEXT:   [2080,   2312,   2544,   2776,   3008,   3240,   3472,   3704,   3936,   4168,   4400,   4632,   4864,   5096,   5328,   5560],
  // CHECK-NEXT:   [2200,   2448,   2696,   2944,   3192,   3440,   3688,   3936,   4184,   4432,   4680,   4928,   5176,   5424,   5672,   5920],
  // CHECK-NEXT:   [2320,   2584,   2848,   3112,   3376,   3640,   3904,   4168,   4432,   4696,   4960,   5224,   5488,   5752,   6016,   6280],
  // CHECK-NEXT:   [2440,   2720,   3000,   3280,   3560,   3840,   4120,   4400,   4680,   4960,   5240,   5520,   5800,   6080,   6360,   6640],
  // CHECK-NEXT:   [2560,   2856,   3152,   3448,   3744,   4040,   4336,   4632,   4928,   5224,   5520,   5816,   6112,   6408,   6704,   7000],
  // CHECK-NEXT:   [2680,   2992,   3304,   3616,   3928,   4240,   4552,   4864,   5176,   5488,   5800,   6112,   6424,   6736,   7048,   7360],
  // CHECK-NEXT:   [2800,   3128,   3456,   3784,   4112,   4440,   4768,   5096,   5424,   5752,   6080,   6408,   6736,   7064,   7392,   7720],
  // CHECK-NEXT:   [2920,   3264,   3608,   3952,   4296,   4640,   4984,   5328,   5672,   6016,   6360,   6704,   7048,   7392,   7736,   8080],
  // CHECK-NEXT:   [3040,   3400,   3760,   4120,   4480,   4840,   5200,   5560,   5920,   6280,   6640,   7000,   7360,   7720,   8080,   8440]
  return
}

func.func private @printMemrefF32(memref<*xf32>)
