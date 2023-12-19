// RUN: mlir-opt %s \
// RUN: | mlir-opt -gpu-lower-to-nvvm="cubin-chip=sm_70 cubin-format=%gpu_compilation_format" \
// RUN: | mlir-cpu-runner \
// RUN:   --shared-libs=%mlir_cuda_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s
// Test case to check the working of Tensor cores on Nvidia GPUs. The kernel has already
// been outlined to prevent crashing due to introduction of an empty basic block by --gpu-
// kernel-outling.
func.func @main() {
  %0 = memref.alloc() : memref<16x16xf16>
  %22 = memref.alloc() : memref<16x16xf16>
  %1 = memref.alloc() : memref<16x16xf32>

  %f1 = arith.constant 1.0e+00 : f16
  %f0 = arith.constant 0.0e+00 : f16
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index

  // Intialize the Input matrix with the column index in each row.
  scf.for %arg0 = %c0 to %c16 step %c1 {
    scf.for %arg1 = %c0 to %c16 step %c1 {
      %2 = arith.index_cast %arg1 : index to i16
      %3 = arith.sitofp %2 : i16 to f16
      memref.store %3, %0[%arg0, %arg1] : memref<16x16xf16>
    }
  }
  // Intialize the accumulator matrix with zeros.
  scf.for %arg0 = %c0 to %c16 step %c1 {
    scf.for %arg1 = %c0 to %c16 step %c1 {
      memref.store %f0, %22[%arg0, %arg1] : memref<16x16xf16>
    }
  }

  %2 = memref.cast %0 : memref<16x16xf16> to memref<*xf16>
  %33 = memref.cast %22 : memref<16x16xf16> to memref<*xf16>
  %3 = memref.cast %1 : memref<16x16xf32> to memref<*xf32>
  gpu.host_register %2 : memref<*xf16>
  gpu.host_register %33 : memref<*xf16>

  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c32, %block_y = %c1, %block_z = %c1) {
    %A = gpu.subgroup_mma_load_matrix %0[%c0, %c0] {leadDimension = 16 : index, transpose} : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "AOp">
    %B = gpu.subgroup_mma_load_matrix %0[%c0, %c0] {leadDimension = 16 : index} : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "BOp">
    %C = gpu.subgroup_mma_load_matrix %22[%c0, %c0] {leadDimension = 16 : index} : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "COp">

    %R = gpu.subgroup_mma_compute %A, %B, %C {a_transpose} : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">

    gpu.subgroup_mma_store_matrix %R, %0[%c0, %c0] {leadDimension = 16 : index}: !gpu.mma_matrix<16x16xf16, "COp">, memref<16x16xf16>
    gpu.terminator
  }

  // Convert the results from f16 to f32 for printing.
  scf.for %arg0 = %c0 to %c16 step %c1 {
    scf.for %arg1 = %c0 to %c16 step %c1 {
      %6 = memref.load %0[%arg0, %arg1] : memref<16x16xf16>
      %7 = arith.extf %6 : f16 to f32
      memref.store %7, %1[%arg0, %arg1] : memref<16x16xf32>
    }
  }

  // Print the memref after computation.
  call @printMemrefF32(%3) : (memref<*xf32>) -> ()
  // CHECK:      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  // CHECK-NEXT: [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240],
  // CHECK-NEXT: [0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480],
  // CHECK-NEXT: [0, 48, 96, 144, 192, 240, 288, 336, 384, 432, 480, 528, 576, 624, 672, 720],
  // CHECK-NEXT: [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960],
  // CHECK-NEXT: [0, 80, 160, 240, 320, 400, 480, 560, 640, 720, 800, 880, 960, 1040, 1120, 1200],
  // CHECK-NEXT: [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440],
  // CHECK-NEXT: [0, 112, 224, 336, 448, 560, 672, 784, 896, 1008, 1120, 1232, 1344, 1456, 1568, 1680],
  // CHECK-NEXT: [0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920],
  // CHECK-NEXT: [0, 144, 288, 432, 576, 720, 864, 1008, 1152, 1296, 1440, 1584, 1728, 1872, 2016, 2160],
  // CHECK-NEXT: [0, 160, 320, 480, 640, 800, 960, 1120, 1280, 1440, 1600, 1760, 1920, 2080, 2240, 2400],
  // CHECK-NEXT: [0, 176, 352, 528, 704, 880, 1056, 1232, 1408, 1584, 1760, 1936, 2112, 2288, 2464, 2640],
  // CHECK-NEXT: [0, 192, 384, 576, 768, 960, 1152, 1344, 1536, 1728, 1920, 2112, 2304, 2496, 2688, 2880],
  // CHECK-NEXT: [0, 208, 416, 624, 832, 1040, 1248, 1456, 1664, 1872, 2080, 2288, 2496, 2704, 2912, 3120],
  // CHECK-NEXT: [0, 224, 448, 672, 896, 1120, 1344, 1568, 1792, 2016, 2240, 2464, 2688, 2912, 3136, 3360],
  // CHECK-NEXT: [0, 240, 480, 720, 960, 1200, 1440, 1680, 1920, 2160, 2400, 2640, 2880, 3120, 3360, 3600]]
  return
}

func.func private @printMemrefF32(memref<*xf32>)
