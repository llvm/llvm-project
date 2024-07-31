// RUN: mlir-opt %s \
// RUN: | mlir-opt -convert-scf-to-cf \
// RUN: | mlir-opt -gpu-kernel-outlining \
// RUN: | mlir-opt -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-rocdl{use-bare-ptr-memref-call-conv=true}),rocdl-attach-target{chip=%chip})' \
// RUN: | mlir-opt -gpu-to-llvm=use-bare-pointers-for-kernels=true -reconcile-unrealized-casts -gpu-module-to-binary \
// RUN: | mlir-cpu-runner \
// RUN:   --shared-libs=%mlir_rocm_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

func.func @vecadd(%arg0 : memref<5xf32>, %arg1 : memref<5xf32>, %arg2 : memref<5xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %block_dim = arith.constant 5 : index
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %block_dim, %block_y = %c1, %block_z = %c1) {
    %a = memref.load %arg0[%tx] : memref<5xf32>
    %b = memref.load %arg1[%tx] : memref<5xf32>
    %c = arith.addf %a, %b : f32
    memref.store %c, %arg2[%tx] : memref<5xf32>
    gpu.terminator
  }
  return
}

// CHECK: [2.46, 2.46, 2.46, 2.46, 2.46]
func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %cf1dot23 = arith.constant 1.23 : f32
  %0 = memref.alloc() : memref<5xf32>
  %1 = memref.alloc() : memref<5xf32>
  %2 = memref.alloc() : memref<5xf32>
  %3 = memref.cast %0 : memref<5xf32> to memref<?xf32>
  %4 = memref.cast %1 : memref<5xf32> to memref<?xf32>
  %5 = memref.cast %2 : memref<5xf32> to memref<?xf32>
  scf.for %i = %c0 to %c5 step %c1 {
    memref.store %cf1dot23, %3[%i] : memref<?xf32>
    memref.store %cf1dot23, %4[%i] : memref<?xf32>
  }
  %6 = memref.cast %3 : memref<?xf32> to memref<*xf32>
  %7 = memref.cast %4 : memref<?xf32> to memref<*xf32>
  %8 = memref.cast %5 : memref<?xf32> to memref<*xf32>
  gpu.host_register %6 : memref<*xf32>
  gpu.host_register %7 : memref<*xf32>
  gpu.host_register %8 : memref<*xf32>
  %9 = call @mgpuMemGetDeviceMemRef1dFloat(%3) : (memref<?xf32>) -> (memref<?xf32>)
  %10 = call @mgpuMemGetDeviceMemRef1dFloat(%4) : (memref<?xf32>) -> (memref<?xf32>)
  %11 = call @mgpuMemGetDeviceMemRef1dFloat(%5) : (memref<?xf32>) -> (memref<?xf32>)
  %12 = memref.cast %9 : memref<?xf32> to memref<5xf32>
  %13 = memref.cast %10 : memref<?xf32> to memref<5xf32>
  %14 = memref.cast %11 : memref<?xf32> to memref<5xf32>

  call @vecadd(%12, %13, %14) : (memref<5xf32>, memref<5xf32>, memref<5xf32>) -> ()
  call @printMemrefF32(%8) : (memref<*xf32>) -> ()
  return
}

func.func private @mgpuMemGetDeviceMemRef1dFloat(%ptr : memref<?xf32>) -> (memref<?xf32>)
func.func private @printMemrefF32(%ptr : memref<*xf32>)
