// RUN: mlir-opt %s \
// RUN:   -gpu-kernel-outlining \
// RUN:   -pass-pipeline='gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin)' \
// RUN:   -gpu-to-llvm \
// RUN: | mlir-cpu-runner \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_cuda_runtime%shlibext \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

func @main() {
  %data = memref.alloc() : memref<2x6xf32>
  %sum = memref.alloc() : memref<2xf32>
  %mul = memref.alloc() : memref<2xf32>
  %cst0 = arith.constant 0.0 : f32
  %cst1 = arith.constant 1.0 : f32
  %cst2 = arith.constant 2.0 : f32
  %cst4 = arith.constant 4.0 : f32
  %cst8 = arith.constant 8.0 : f32
  %cst16 = arith.constant 16.0 : f32

  %cst3 = arith.constant 3.0 : f32
  %cst6 = arith.constant 6.0 : f32
  %cst7 = arith.constant 7.0 : f32
  %cst10 = arith.constant 10.0 : f32
  %cst11 = arith.constant 11.0 : f32

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index

  %cast_data = memref.cast %data : memref<2x6xf32> to memref<*xf32>
  gpu.host_register %cast_data : memref<*xf32>
  %cast_sum = memref.cast %sum : memref<2xf32> to memref<*xf32>
  gpu.host_register %cast_sum : memref<*xf32>
  %cast_mul = memref.cast %mul : memref<2xf32> to memref<*xf32>
  gpu.host_register %cast_mul : memref<*xf32>

  memref.store %cst0, %data[%c0, %c0] : memref<2x6xf32>
  memref.store %cst1, %data[%c0, %c1] : memref<2x6xf32>
  memref.store %cst2, %data[%c0, %c2] : memref<2x6xf32>
  memref.store %cst4, %data[%c0, %c3] : memref<2x6xf32>
  memref.store %cst8, %data[%c0, %c4] : memref<2x6xf32>
  memref.store %cst16, %data[%c0, %c5] : memref<2x6xf32>

  memref.store %cst2, %data[%c1, %c0] : memref<2x6xf32>
  memref.store %cst3, %data[%c1, %c1] : memref<2x6xf32>
  memref.store %cst6, %data[%c1, %c2] : memref<2x6xf32>
  memref.store %cst7, %data[%c1, %c3] : memref<2x6xf32>
  memref.store %cst10, %data[%c1, %c4] : memref<2x6xf32>
  memref.store %cst11, %data[%c1, %c5] : memref<2x6xf32>

  // ADD + MUL
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c2, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c6, %block_y = %c1, %block_z = %c1) {
    %val = memref.load %data[%bx, %tx] : memref<2x6xf32>
    %reduced0 = "gpu.all_reduce"(%val) ({}) { op = "add" } : (f32) -> (f32)
    memref.store %reduced0, %sum[%bx] : memref<2xf32>
    %reduced1 = "gpu.all_reduce"(%val) ({}) { op = "mul" } : (f32) -> (f32)
    memref.store %reduced1, %mul[%bx] : memref<2xf32>
    gpu.terminator
  }

  call @print_memref_f32(%cast_sum) : (memref<*xf32>) -> ()
  // CHECK: [31, 39]

  call @print_memref_f32(%cast_mul) : (memref<*xf32>) -> ()
  // CHECK: [0, 27720]

  return
}

func private @print_memref_f32(memref<*xf32>)
