// RUN: mlir-opt %s \
// RUN:   -gpu-kernel-outlining \
// RUN:   -pass-pipeline='gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin)' \
// RUN:   -gpu-to-llvm \
// RUN: | mlir-cpu-runner \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_cuda_runtime%shlibext \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

func @other_func(%arg0 : f32, %arg1 : memref<?xf32>) {
  %cst = constant 1 : index
  %c0 = constant 0 : index
  %cst2 = memref.dim %arg1, %c0 : memref<?xf32>
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %cst, %grid_y = %cst, %grid_z = %cst)
             threads(%tx, %ty, %tz) in (%block_x = %cst2, %block_y = %cst, %block_z = %cst) {
    memref.store %arg0, %arg1[%tx] : memref<?xf32>
    gpu.terminator
  }
  return
}

// CHECK: [1, 1, 1, 1, 1]
// CHECK: ( 1, 1 )
func @main() {
  %v0 = constant 0.0 : f32
  %c0 = constant 0: index
  %arg0 = memref.alloc() : memref<5xf32>
  %21 = constant 5 : i32
  %22 = memref.cast %arg0 : memref<5xf32> to memref<?xf32>
  %23 = memref.cast %22 : memref<?xf32> to memref<*xf32>
  gpu.host_register %23 : memref<*xf32>
  call @print_memref_f32(%23) : (memref<*xf32>) -> ()
  %24 = constant 1.0 : f32
  call @other_func(%24, %22) : (f32, memref<?xf32>) -> ()
  call @print_memref_f32(%23) : (memref<*xf32>) -> ()
  %val1 = vector.transfer_read %arg0[%c0], %v0: memref<5xf32>, vector<2xf32>
  vector.print %val1: vector<2xf32>
  return
}

func private @print_memref_f32(%ptr : memref<*xf32>)
