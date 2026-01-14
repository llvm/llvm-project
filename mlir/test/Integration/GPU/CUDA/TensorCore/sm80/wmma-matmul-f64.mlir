// RUN: mlir-opt %s \
// RUN: | mlir-opt -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_80 cubin-format=%gpu_compilation_format" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_cuda_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d1, d0)>

func.func @main() {
  %a = memref.alloc() : memref<8x4xf64>
  %b = memref.alloc() : memref<4x8xf64>
  %c = memref.alloc() : memref<8x8xf64>
  %d = memref.alloc() : memref<8x8xf64>

  %f1 = arith.constant 1.0e+00 : f64
  %fcst = arith.constant 3.14e+00 : f64
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index

  // Initialize the Input matrixes with ones.
  scf.for %arg0 = %c0 to %c8 step %c1 {
    scf.for %arg1 = %c0 to %c4 step %c1 {
      memref.store %f1, %a[%arg0, %arg1] : memref<8x4xf64>
      memref.store %f1, %b[%arg1, %arg0] : memref<4x8xf64>
    }
  }
  // Initialize the accumulator matrix with a constant.
  scf.for %arg0 = %c0 to %c8 step %c1 {
    scf.for %arg1 = %c0 to %c8 step %c1 {
      memref.store %fcst, %c[%arg0, %arg1] : memref<8x8xf64>
    }
  }

  %2 = memref.cast %a : memref<8x4xf64> to memref<*xf64>
  %20 = memref.cast %b : memref<4x8xf64> to memref<*xf64>
  %33 = memref.cast %c : memref<8x8xf64> to memref<*xf64>
  %34 = memref.cast %d : memref<8x8xf64> to memref<*xf64>

  gpu.host_register %2 : memref<*xf64>
  gpu.host_register %20 : memref<*xf64>
  gpu.host_register %33 : memref<*xf64>
  gpu.host_register %34 : memref<*xf64>

  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c32, %block_y = %c1, %block_z = %c1) {
    %A = gpu.subgroup_mma_load_matrix %a[%c0, %c0] {leadDimension = 4 : index} : memref<8x4xf64> -> !gpu.mma_matrix<8x4xf64, "AOp">
    %B = gpu.subgroup_mma_load_matrix %b[%c0, %c0] {leadDimension = 8 : index} : memref<4x8xf64> -> !gpu.mma_matrix<4x8xf64, "BOp">
    %C = gpu.subgroup_mma_load_matrix %c[%c0, %c0] {leadDimension = 8 : index} : memref<8x8xf64> -> !gpu.mma_matrix<8x8xf64, "COp">

    %R = gpu.subgroup_mma_compute %A, %B, %C : !gpu.mma_matrix<8x4xf64, "AOp">, !gpu.mma_matrix<4x8xf64, "BOp"> -> !gpu.mma_matrix<8x8xf64, "COp">

    gpu.subgroup_mma_store_matrix %R, %d[%c0, %c0] {leadDimension = 8 : index}: !gpu.mma_matrix<8x8xf64, "COp">, memref<8x8xf64>
    gpu.terminator
  }
  // Print the memref after computation.
  call @printMemrefF64(%34) : (memref<*xf64>) -> ()
  // CHECK: [7.14,   7.14,   7.14,   7.14,   7.14,   7.14,   7.14,   7.14],
  // CHECK-NEXT: [7.14,   7.14,   7.14,   7.14,   7.14,   7.14,   7.14,   7.14],
  // CHECK-NEXT: [7.14,   7.14,   7.14,   7.14,   7.14,   7.14,   7.14,   7.14],
  // CHECK-NEXT: [7.14,   7.14,   7.14,   7.14,   7.14,   7.14,   7.14,   7.14],
  // CHECK-NEXT: [7.14,   7.14,   7.14,   7.14,   7.14,   7.14,   7.14,   7.14],
  // CHECK-NEXT: [7.14,   7.14,   7.14,   7.14,   7.14,   7.14,   7.14,   7.14],
  // CHECK-NEXT: [7.14,   7.14,   7.14,   7.14,   7.14,   7.14,   7.14,   7.14],
  // CHECK-NEXT: [7.14,   7.14,   7.14,   7.14,   7.14,   7.14,   7.14,   7.14]
  return
}

func.func private @printMemrefF64(memref<*xf64>)
