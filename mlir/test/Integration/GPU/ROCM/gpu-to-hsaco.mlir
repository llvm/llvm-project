// RUN: mlir-opt %s \
// RUN: | mlir-opt -gpu-kernel-outlining \
// RUN: | mlir-opt -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-rocdl),rocdl-attach-target{chip=%chip})' \
// RUN: | mlir-opt -gpu-to-llvm -reconcile-unrealized-casts -gpu-module-to-binary \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_rocm_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

func.func @other_func(%arg0 : f32, %arg1 : memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %block_dim = memref.dim %arg1, %c0 : memref<?xf32>
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %block_dim, %block_y = %c1, %block_z = %c1) {
    memref.store %arg0, %arg1[%tx] : memref<?xf32>
    gpu.terminator
  }
  return
}

// CHECK: [1, 1, 1, 1, 1]
func.func @main() {
  %arg0 = memref.alloc() : memref<5xf32>
  %21 = arith.constant 5 : i32
  %22 = memref.cast %arg0 : memref<5xf32> to memref<?xf32>
  %cast = memref.cast %22 : memref<?xf32> to memref<*xf32>
  gpu.host_register %cast : memref<*xf32>
  %23 = memref.cast %22 : memref<?xf32> to memref<*xf32>
  call @printMemrefF32(%23) : (memref<*xf32>) -> ()
  %24 = arith.constant 1.0 : f32
  %25 = call @mgpuMemGetDeviceMemRef1dFloat(%22) : (memref<?xf32>) -> (memref<?xf32>)
  call @other_func(%24, %25) : (f32, memref<?xf32>) -> ()
  call @printMemrefF32(%23) : (memref<*xf32>) -> ()
  return
}

func.func private @mgpuMemGetDeviceMemRef1dFloat(%ptr : memref<?xf32>) -> (memref<?xf32>)
func.func private @printMemrefF32(%ptr : memref<*xf32>)
