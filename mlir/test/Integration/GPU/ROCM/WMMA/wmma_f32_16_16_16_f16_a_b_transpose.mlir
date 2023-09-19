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
      %float = arith.sitofp %cast_c : i16 to f16
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
    %A = gpu.subgroup_mma_load_matrix %gpu_in[%c0, %c0] {leadDimension = 16 : index, transpose} : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "AOp">
    %B = gpu.subgroup_mma_load_matrix %gpu_in[%c0, %c0] {leadDimension = 16 : index, transpose} : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "BOp">
    %C = gpu.subgroup_mma_load_matrix %gpu_out[%c0, %c0] {leadDimension = 16 : index} : memref<16x16xf32> -> !gpu.mma_matrix<16x16xf32, "COp">

    %R = gpu.subgroup_mma_compute %A, %B, %C : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">

    gpu.subgroup_mma_store_matrix %R, %gpu_out[%c0, %c0] {leadDimension = 16 : index}: !gpu.mma_matrix<16x16xf32, "COp">, memref<16x16xf32>
    gpu.terminator
  }

  %asyncToken_4 = gpu.memcpy async [%stream] %22, %gpu_out : memref<16x16xf32>, memref<16x16xf32>
  gpu.wait [%stream]

  // Print the memref after computation.
  call @printMemrefF32(%33) : (memref<*xf32>) -> ()
  // CHECK:      [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
  // CHECK-NEXT: [120,   120,   120,   120,   120,   120,   120,   120,   120,   120,   120,   120,   120,   120,   120,   120],
  // CHECK-NEXT: [240,   240,   240,   240,   240,   240,   240,   240,   240,   240,   240,   240,   240,   240,   240,   240],
  // CHECK-NEXT: [360,   360,   360,   360,   360,   360,   360,   360,   360,   360,   360,   360,   360,   360,   360,   360],
  // CHECK-NEXT: [480,   480,   480,   480,   480,   480,   480,   480,   480,   480,   480,   480,   480,   480,   480,   480],
  // CHECK-NEXT: [600,   600,   600,   600,   600,   600,   600,   600,   600,   600,   600,   600,   600,   600,   600,   600],
  // CHECK-NEXT: [720,   720,   720,   720,   720,   720,   720,   720,   720,   720,   720,   720,   720,   720,   720,   720],
  // CHECK-NEXT: [840,   840,   840,   840,   840,   840,   840,   840,   840,   840,   840,   840,   840,   840,   840,   840],
  // CHECK-NEXT: [960,   960,   960,   960,   960,   960,   960,   960,   960,   960,   960,   960,   960,   960,   960,   960],
  // CHECK-NEXT: [1080,   1080,   1080,   1080,   1080,   1080,   1080,   1080,   1080,   1080,   1080,   1080,   1080,   1080,   1080,   1080],
  // CHECK-NEXT: [1200,   1200,   1200,   1200,   1200,   1200,   1200,   1200,   1200,   1200,   1200,   1200,   1200,   1200,   1200,   1200],
  // CHECK-NEXT: [1320,   1320,   1320,   1320,   1320,   1320,   1320,   1320,   1320,   1320,   1320,   1320,   1320,   1320,   1320,   1320],
  // CHECK-NEXT: [1440,   1440,   1440,   1440,   1440,   1440,   1440,   1440,   1440,   1440,   1440,   1440,   1440,   1440,   1440,   1440],
  // CHECK-NEXT: [1560,   1560,   1560,   1560,   1560,   1560,   1560,   1560,   1560,   1560,   1560,   1560,   1560,   1560,   1560,   1560],
  // CHECK-NEXT: [1680,   1680,   1680,   1680,   1680,   1680,   1680,   1680,   1680,   1680,   1680,   1680,   1680,   1680,   1680,   1680],
  // CHECK-NEXT: [1800,   1800,   1800,   1800,   1800,   1800,   1800,   1800,   1800,   1800,   1800,   1800,   1800,   1800,   1800,   1800]
  return
}

func.func private @printMemrefF32(memref<*xf32>)
