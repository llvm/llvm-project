// RUN: mlir-opt %s \
// RUN:   -gpu-kernel-outlining \
// RUN:   -pass-pipeline='gpu.module(strip-debuginfo,convert-gpu-to-rocdl,gpu-to-hsaco)' \
// RUN:   -gpu-to-llvm \
// RUN: | mlir-cpu-runner \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

func @other_func(%arg0 : f32, %arg1 : memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %block_dim = dim %arg1, %c0 : memref<?xf32>
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %block_dim, %block_y = %c1, %block_z = %c1) {
    store %arg0, %arg1[%tx] : memref<?xf32>
    gpu.terminator
  }
  return
}

// CHECK: [1, 1, 1, 1, 1]
func @main() {
  %arg0 = alloc() : memref<5xf32>
  %21 = arith.constant 5 : i32
  %22 = memref_cast %arg0 : memref<5xf32> to memref<?xf32>
  %cast = memref_cast %22 : memref<?xf32> to memref<*xf32>
  gpu.host_register %cast : memref<*xf32>
  %23 = memref_cast %22 : memref<?xf32> to memref<*xf32>
  call @print_memref_f32(%23) : (memref<*xf32>) -> ()
  %24 = arith.constant 1.0 : f32
  %25 = call @mgpuMemGetDeviceMemRef1dFloat(%22) : (memref<?xf32>) -> (memref<?xf32>)
  call @other_func(%24, %25) : (f32, memref<?xf32>) -> ()
  call @print_memref_f32(%23) : (memref<*xf32>) -> ()
  return
}

func private @mgpuMemGetDeviceMemRef1dFloat(%ptr : memref<?xf32>) -> (memref<?xf32>)
func private @print_memref_f32(%ptr : memref<*xf32>)
