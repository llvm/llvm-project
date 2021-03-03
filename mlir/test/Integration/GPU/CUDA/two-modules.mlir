// RUN: mlir-cuda-runner %s \
// RUN:   -gpu-to-cubin="gpu-binary-annotation=nvvm.cubin" \
// RUN:   -gpu-to-llvm="gpu-binary-annotation=nvvm.cubin" \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_cuda_runtime%shlibext \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// CHECK: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
func @main() {
  %arg = alloc() : memref<13xi32>
  %dst = memref_cast %arg : memref<13xi32> to memref<?xi32>
  %one = constant 1 : index
  %c0 = constant 0 : index
  %sx = dim %dst, %c0 : memref<?xi32>
  %cast_dst = memref_cast %dst : memref<?xi32> to memref<*xi32>
  gpu.host_register %cast_dst : memref<*xi32>
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %one, %grid_y = %one, %grid_z = %one)
             threads(%tx, %ty, %tz) in (%block_x = %sx, %block_y = %one, %block_z = %one) {
    %t0 = index_cast %tx : index to i32
    store %t0, %dst[%tx] : memref<?xi32>
    gpu.terminator
  }
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %one, %grid_y = %one, %grid_z = %one)
             threads(%tx, %ty, %tz) in (%block_x = %sx, %block_y = %one, %block_z = %one) {
    %t0 = index_cast %tx : index to i32
    store %t0, %dst[%tx] : memref<?xi32>
    gpu.terminator
  }
  call @print_memref_i32(%cast_dst) : (memref<*xi32>) -> ()
  return
}

func private @print_memref_i32(%memref : memref<*xi32>)
