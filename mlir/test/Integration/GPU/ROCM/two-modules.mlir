// RUN: mlir-opt %s \
// RUN:   -gpu-kernel-outlining \
// RUN:   -pass-pipeline='gpu.module(strip-debuginfo,convert-gpu-to-rocdl,gpu-to-hsaco{chip=%chip})' \
// RUN:   -gpu-to-llvm \
// RUN: | mlir-cpu-runner \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// CHECK: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
func @main() {
  %arg = memref.alloc() : memref<13xi32>
  %dst = memref.cast %arg : memref<13xi32> to memref<?xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %sx = memref.dim %dst, %c0 : memref<?xi32>
  %cast_dst = memref.cast %dst : memref<?xi32> to memref<*xi32>
  gpu.host_register %cast_dst : memref<*xi32>
  %dst_device = call @mgpuMemGetDeviceMemRef1dInt32(%dst) : (memref<?xi32>) -> (memref<?xi32>)
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %sx, %block_y = %c1, %block_z = %c1) {
    %t0 = arith.index_cast %tx : index to i32
    memref.store %t0, %dst_device[%tx] : memref<?xi32>
    gpu.terminator
  }
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %sx, %block_y = %c1, %block_z = %c1) {
    %t0 = arith.index_cast %tx : index to i32
    memref.store %t0, %dst_device[%tx] : memref<?xi32>
    gpu.terminator
  }
  call @print_memref_i32(%cast_dst) : (memref<*xi32>) -> ()
  return
}

func private @mgpuMemGetDeviceMemRef1dInt32(%ptr : memref<?xi32>) -> (memref<?xi32>)
func private @print_memref_i32(%ptr : memref<*xi32>)
