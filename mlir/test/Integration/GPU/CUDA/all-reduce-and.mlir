// RUN: mlir-cuda-runner %s \
// RUN:   -gpu-to-cubin="gpu-binary-annotation=nvvm.cubin" \
// RUN:   -gpu-to-llvm="gpu-binary-annotation=nvvm.cubin" \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_cuda_runtime%shlibext \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

func @main() {
  %data = alloc() : memref<2x6xi32>
  %sum = alloc() : memref<2xi32>
  %cst0 = constant 0 : i32
  %cst1 = constant 1 : i32
  %cst2 = constant 2 : i32
  %cst4 = constant 4 : i32
  %cst8 = constant 8 : i32
  %cst16 = constant 16 : i32

  %cst3 = constant 3 : i32
  %cst6 = constant 6 : i32
  %cst7 = constant 7 : i32
  %cst10 = constant 10 : i32
  %cst11 = constant 11 : i32

  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %c5 = constant 5 : index
  %c6 = constant 6 : index

  %cast_data = memref_cast %data : memref<2x6xi32> to memref<*xi32>
  gpu.host_register %cast_data : memref<*xi32>
  %cast_sum = memref_cast %sum : memref<2xi32> to memref<*xi32>
  gpu.host_register %cast_sum : memref<*xi32>

  store %cst0, %data[%c0, %c0] : memref<2x6xi32>
  store %cst1, %data[%c0, %c1] : memref<2x6xi32>
  store %cst2, %data[%c0, %c2] : memref<2x6xi32>
  store %cst4, %data[%c0, %c3] : memref<2x6xi32>
  store %cst8, %data[%c0, %c4] : memref<2x6xi32>
  store %cst16, %data[%c0, %c5] : memref<2x6xi32>

  store %cst2, %data[%c1, %c0] : memref<2x6xi32>
  store %cst3, %data[%c1, %c1] : memref<2x6xi32>
  store %cst6, %data[%c1, %c2] : memref<2x6xi32>
  store %cst7, %data[%c1, %c3] : memref<2x6xi32>
  store %cst10, %data[%c1, %c4] : memref<2x6xi32>
  store %cst11, %data[%c1, %c5] : memref<2x6xi32>

  // AND
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c2, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c6, %block_y = %c1, %block_z = %c1) {
    %val = load %data[%bx, %tx] : memref<2x6xi32>
    %reduced = "gpu.all_reduce"(%val) ({}) { op = "and" } : (i32) -> (i32)
    store %reduced, %sum[%bx] : memref<2xi32>
    gpu.terminator
  }

  call @print_memref_i32(%cast_sum) : (memref<*xi32>) -> ()
  // CHECK: [0, 2]

  return
}

func private @print_memref_i32(memref<*xi32>)

