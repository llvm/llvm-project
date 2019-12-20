// RUN: mlir-cuda-runner %s --shared-libs=%cuda_wrapper_library_dir/libcuda-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// CHECK: [4, 5, 6, 7, 0, 1, 2, 3, 12, -1, -1, -1, 8]
func @main() {
  %arg = alloc() : memref<13xf32>
  %dst = memref_cast %arg : memref<13xf32> to memref<?xf32>
  %one = constant 1 : index
  %sx = dim %dst, 0 : memref<?xf32>
  call @mcuMemHostRegisterMemRef1dFloat(%dst) : (memref<?xf32>) -> ()
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %one, %grid_y = %one, %grid_z = %one)
             threads(%tx, %ty, %tz) in (%block_x = %sx, %block_y = %one, %block_z = %one)
             args(%kernel_dst = %dst) : memref<?xf32> {
    %t0 = index_cast %tx : index to i32
    %val = sitofp %t0 : i32 to f32
    %width = index_cast %block_x : index to i32
    %offset = constant 4 : i32
    %shfl, %valid = gpu.shuffle %val, %offset, %width xor : f32
    cond_br %valid, ^bb1(%shfl : f32), ^bb0
  ^bb0:
    %m1 = constant -1.0 : f32
    br ^bb1(%m1 : f32)
  ^bb1(%value : f32):
    store %value, %kernel_dst[%tx] : memref<?xf32>
    gpu.return
  }
  %U = memref_cast %dst : memref<?xf32> to memref<*xf32>
  call @print_memref_f32(%U) : (memref<*xf32>) -> ()
  return
}

func @mcuMemHostRegisterMemRef1dFloat(%ptr : memref<?xf32>)
func @print_memref_f32(%ptr : memref<*xf32>)
