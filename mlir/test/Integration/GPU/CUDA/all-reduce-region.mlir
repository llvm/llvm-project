// RUN: mlir-opt %s \
// RUN:   -gpu-kernel-outlining \
// RUN:   -pass-pipeline='gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin)' \
// RUN:   -gpu-to-llvm \
// RUN: | mlir-cpu-runner \
// RUN:   --shared-libs=%mlir_lib_dir/libmlir_cuda_runtime%shlibext \
// RUN:   --shared-libs=%mlir_lib_dir/libmlir_runner_utils%shlibext \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// CHECK: [{{(35, ){34}35}}]
func.func @main() {
  %arg = memref.alloc() : memref<35xf32>
  %dst = memref.cast %arg : memref<35xf32> to memref<?xf32>
  %one = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %sx = memref.dim %dst, %c0 : memref<?xf32>
  %cast_dst = memref.cast %dst : memref<?xf32> to memref<*xf32>
  gpu.host_register %cast_dst : memref<*xf32>
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %one, %grid_y = %one, %grid_z = %one)
             threads(%tx, %ty, %tz) in (%block_x = %sx, %block_y = %one, %block_z = %one) {
    %val = arith.index_cast %tx : index to i32
    %xor = gpu.all_reduce %val {
    ^bb(%lhs : i32, %rhs : i32):
      %xor = arith.xori %lhs, %rhs : i32
      "gpu.yield"(%xor) : (i32) -> ()
    } : (i32) -> (i32)
    %res = arith.sitofp %xor : i32 to f32
    memref.store %res, %dst[%tx] : memref<?xf32>
    gpu.terminator
  }
  call @printMemrefF32(%cast_dst) : (memref<*xf32>) -> ()
  return
}

func.func private @printMemrefF32(memref<*xf32>)
