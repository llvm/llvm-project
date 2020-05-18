// RUN: mlir-cuda-runner %s --shared-libs=%cuda_wrapper_library_dir/libcuda-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// CHECK-COUNT-8: [{{(5356, ){12}5356}}]
func @main() {
  %arg = alloc() : memref<2x4x13xf32>
  %dst = memref_cast %arg : memref<2x4x13xf32> to memref<?x?x?xf32>
  %one = constant 1 : index
  %sx = dim %dst, 2 : memref<?x?x?xf32>
  %sy = dim %dst, 1 : memref<?x?x?xf32>
  %sz = dim %dst, 0 : memref<?x?x?xf32>
  %cast_dst = memref_cast %dst : memref<?x?x?xf32> to memref<*xf32>
  call @mcuMemHostRegisterFloat(%cast_dst) : (memref<*xf32>) -> ()
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %one, %grid_y = %one, %grid_z = %one)
             threads(%tx, %ty, %tz) in (%block_x = %sx, %block_y = %sy, %block_z = %sz) {
    %t0 = muli %tz, %block_y : index
    %t1 = addi %ty, %t0 : index
    %t2 = muli %t1, %block_x : index
    %idx = addi %tx, %t2 : index
    %t3 = index_cast %idx : index to i32
    %val = sitofp %t3 : i32 to f32
    %sum = "gpu.all_reduce"(%val) ({}) { op = "add" } : (f32) -> (f32)
    store %sum, %dst[%tz, %ty, %tx] : memref<?x?x?xf32>
    gpu.terminator
  }
  call @print_memref_f32(%cast_dst) : (memref<*xf32>) -> ()
  return
}

func @mcuMemHostRegisterFloat(%ptr : memref<*xf32>)
func @print_memref_f32(%ptr : memref<*xf32>)
