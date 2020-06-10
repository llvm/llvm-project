// RUN: mlir-cuda-runner %s --shared-libs=%cuda_wrapper_library_dir/libcuda-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

func @other_func(%arg0 : f32, %arg1 : memref<?xf32>) {
  %cst = constant 1 : index
  %c0 = constant 0 : index
  %cst2 = dim %arg1, %c0 : memref<?xf32>
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %cst, %grid_y = %cst, %grid_z = %cst)
             threads(%tx, %ty, %tz) in (%block_x = %cst2, %block_y = %cst, %block_z = %cst) {
    store %arg0, %arg1[%tx] : memref<?xf32>
    gpu.terminator
  }
  return
}

// CHECK: [1, 1, 1, 1, 1]
func @main() {
  %arg0 = alloc() : memref<5xf32>
  %21 = constant 5 : i32
  %22 = memref_cast %arg0 : memref<5xf32> to memref<?xf32>
  %23 = memref_cast %22 : memref<?xf32> to memref<*xf32>
  call @mcuMemHostRegisterFloat(%23) : (memref<*xf32>) -> ()
  call @print_memref_f32(%23) : (memref<*xf32>) -> ()
  %24 = constant 1.0 : f32
  call @other_func(%24, %22) : (f32, memref<?xf32>) -> ()
  call @print_memref_f32(%23) : (memref<*xf32>) -> ()
  return
}

func @mcuMemHostRegisterFloat(%ptr : memref<*xf32>)
func @print_memref_f32(%ptr : memref<*xf32>)
