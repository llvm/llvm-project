// RUN: mlir-opt %s \
// RUN: | mlir-opt -gpu-kernel-outlining \
// RUN: | mlir-opt -pass-pipeline='gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin)' \
// RUN: | mlir-opt -gpu-to-llvm \
// RUN: | mlir-cpu-runner \
// RUN:   --shared-libs=%mlir_lib_dir/libmlir_cuda_runtime%shlibext \
// RUN:   --shared-libs=%mlir_lib_dir/libmlir_runner_utils%shlibext \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// CHECK: [4, 5, 6, 7, 0, 1, 2, 3, 12, -1, -1, -1, 8]
func.func @main() {
  %arg = memref.alloc() : memref<13xf32>
  %dst = memref.cast %arg : memref<13xf32> to memref<?xf32>
  %one = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %sx = memref.dim %dst, %c0 : memref<?xf32>
  %cast_dst = memref.cast %dst : memref<?xf32> to memref<*xf32>
  gpu.host_register %cast_dst : memref<*xf32>
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %one, %grid_y = %one, %grid_z = %one)
             threads(%tx, %ty, %tz) in (%block_x = %sx, %block_y = %one, %block_z = %one) {
    %t0 = arith.index_cast %tx : index to i32
    %val = arith.sitofp %t0 : i32 to f32
    %width = arith.index_cast %block_x : index to i32
    %offset = arith.constant 4 : i32
    %shfl, %valid = gpu.shuffle xor %val, %offset, %width : f32
    cf.cond_br %valid, ^bb1(%shfl : f32), ^bb0
  ^bb0:
    %m1 = arith.constant -1.0 : f32
    cf.br ^bb1(%m1 : f32)
  ^bb1(%value : f32):
    memref.store %value, %dst[%tx] : memref<?xf32>
    gpu.terminator
  }
  call @printMemrefF32(%cast_dst) : (memref<*xf32>) -> ()
  return
}

func.func private @printMemrefF32(%ptr : memref<*xf32>)
