// RUN: mlir-opt %s \
// RUN: | mlir-opt -gpu-kernel-outlining \
// RUN: | mlir-opt -pass-pipeline='gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin)' \
// RUN: | mlir-opt -gpu-to-llvm \
// RUN: | mlir-cpu-runner \
// RUN:   --shared-libs=%mlir_lib_dir/libmlir_cuda_runtime%shlibext \
// RUN:   --shared-libs=%mlir_lib_dir/libmlir_runner_utils%shlibext \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// CHECK: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
func.func @main() {
  %arg = memref.alloc() : memref<13xi32>
  %dst = memref.cast %arg : memref<13xi32> to memref<?xi32>
  %one = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %sx = memref.dim %dst, %c0 : memref<?xi32>
  %cast_dst = memref.cast %dst : memref<?xi32> to memref<*xi32>
  gpu.host_register %cast_dst : memref<*xi32>
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %one, %grid_y = %one, %grid_z = %one)
             threads(%tx, %ty, %tz) in (%block_x = %sx, %block_y = %one, %block_z = %one) {
    %t0 = arith.index_cast %tx : index to i32
    memref.store %t0, %dst[%tx] : memref<?xi32>
    gpu.terminator
  }
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %one, %grid_y = %one, %grid_z = %one)
             threads(%tx, %ty, %tz) in (%block_x = %sx, %block_y = %one, %block_z = %one) {
    %t0 = arith.index_cast %tx : index to i32
    memref.store %t0, %dst[%tx] : memref<?xi32>
    gpu.terminator
  }
  call @printMemrefI32(%cast_dst) : (memref<*xi32>) -> ()
  return
}

func.func private @printMemrefI32(%memref : memref<*xi32>)
