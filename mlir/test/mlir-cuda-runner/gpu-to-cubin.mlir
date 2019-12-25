// RUN: mlir-cuda-runner %s --shared-libs=%cuda_wrapper_library_dir/libcuda-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

func @other_func(%arg0 : f32, %arg1 : memref<?xf32>) {
  %cst = constant 1 : index
  %cst2 = dim %arg1, 0 : memref<?xf32>
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %cst, %grid_y = %cst, %grid_z = %cst)
             threads(%tx, %ty, %tz) in (%block_x = %cst2, %block_y = %cst, %block_z = %cst)
             args(%kernel_arg0 = %arg0, %kernel_arg1 = %arg1) : f32, memref<?xf32> {
    store %kernel_arg0, %kernel_arg1[%tx] : memref<?xf32>
    gpu.return
  }
  return
}

// CHECK: [1, 1, 1, 1, 1]
func @main() {
  %arg0 = alloc() : memref<5xf32>
  %21 = constant 5 : i32
  %22 = memref_cast %arg0 : memref<5xf32> to memref<?xf32>
  call @mcuMemHostRegisterMemRef1dFloat(%22) : (memref<?xf32>) -> ()
  call @print_memref_1d_f32(%22) : (memref<?xf32>) -> ()
  %24 = constant 1.0 : f32
  call @other_func(%24, %22) : (f32, memref<?xf32>) -> ()
  call @print_memref_1d_f32(%22) : (memref<?xf32>) -> ()
  return
}

func @mcuMemHostRegisterMemRef1dFloat(%ptr : memref<?xf32>)
func @print_memref_1d_f32(memref<?xf32>)
