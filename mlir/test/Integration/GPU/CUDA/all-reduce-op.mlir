// RUN: mlir-opt %s \
// RUN: | mlir-opt -gpu-kernel-outlining \
// RUN: | mlir-opt -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin))' \
// RUN: | mlir-opt -gpu-to-llvm \
// RUN: | mlir-cpu-runner \
// RUN:   --shared-libs=%mlir_lib_dir/libmlir_cuda_runtime%shlibext \
// RUN:   --shared-libs=%mlir_lib_dir/libmlir_runner_utils%shlibext \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// CHECK-COUNT-8: [{{(5356, ){12}5356}}]
func.func @main() {
  %arg = memref.alloc() : memref<2x4x13xf32>
  %dst = memref.cast %arg : memref<2x4x13xf32> to memref<?x?x?xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %sx = memref.dim %dst, %c2 : memref<?x?x?xf32>
  %sy = memref.dim %dst, %c1 : memref<?x?x?xf32>
  %sz = memref.dim %dst, %c0 : memref<?x?x?xf32>
  %cast_dst = memref.cast %dst : memref<?x?x?xf32> to memref<*xf32>
  gpu.host_register %cast_dst : memref<*xf32>
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %sx, %block_y = %sy, %block_z = %sz) {
    %t0 = arith.muli %tz, %block_y : index
    %t1 = arith.addi %ty, %t0 : index
    %t2 = arith.muli %t1, %block_x : index
    %idx = arith.addi %tx, %t2 : index
    %t3 = arith.index_cast %idx : index to i32
    %val = arith.sitofp %t3 : i32 to f32
    %sum = gpu.all_reduce add %val uniform {} : (f32) -> (f32)
    memref.store %sum, %dst[%tz, %ty, %tx] : memref<?x?x?xf32>
    gpu.terminator
  }
  call @printMemrefF32(%cast_dst) : (memref<*xf32>) -> ()
  return
}

func.func private @printMemrefF32(%ptr : memref<*xf32>)
