// RUN: mlir-cuda-runner %s \
// RUN:   --shared-libs=%cuda_wrapper_library_dir/libcuda-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// CHECK: [{{(35, ){34}35}}]
func @main() {
  %arg = alloc() : memref<35xf32>
  %dst = memref_cast %arg : memref<35xf32> to memref<?xf32>
  %one = constant 1 : index
  %c0 = constant 0 : index
  %sx = dim %dst, %c0 : memref<?xf32>
  %cast_dst = memref_cast %dst : memref<?xf32> to memref<*xf32>
  gpu.host_register %cast_dst : memref<*xf32>
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %one, %grid_y = %one, %grid_z = %one)
             threads(%tx, %ty, %tz) in (%block_x = %sx, %block_y = %one, %block_z = %one) {
    %val = index_cast %tx : index to i32
    %xor = "gpu.all_reduce"(%val) ({
    ^bb(%lhs : i32, %rhs : i32):
      %xor = xor %lhs, %rhs : i32
      "gpu.yield"(%xor) : (i32) -> ()
    }) : (i32) -> (i32)
    %res = sitofp %xor : i32 to f32
    store %res, %dst[%tx] : memref<?xf32>
    gpu.terminator
  }
  call @print_memref_f32(%cast_dst) : (memref<*xf32>) -> ()
  return
}

func private @print_memref_f32(memref<*xf32>)
