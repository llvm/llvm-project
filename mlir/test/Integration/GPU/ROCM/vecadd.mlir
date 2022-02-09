// RUN: mlir-opt %s \
// RUN:   -convert-scf-to-cf \
// RUN:   -gpu-kernel-outlining \
// RUN:   -pass-pipeline='gpu.module(strip-debuginfo,convert-gpu-to-rocdl,gpu-to-hsaco{chip=%chip})' \
// RUN:   -gpu-to-llvm \
// RUN: | mlir-cpu-runner \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

func @vecadd(%arg0 : memref<?xf32>, %arg1 : memref<?xf32>, %arg2 : memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %block_dim = memref.dim %arg0, %c0 : memref<?xf32>
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %block_dim, %block_y = %c1, %block_z = %c1) {
    %a = memref.load %arg0[%tx] : memref<?xf32>
    %b = memref.load %arg1[%tx] : memref<?xf32>
    %c = arith.addf %a, %b : f32
    memref.store %c, %arg2[%tx] : memref<?xf32>
    gpu.terminator
  }
  return
}

// CHECK: [2.46, 2.46, 2.46, 2.46, 2.46]
func @main() {
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

  call @vecadd(%9, %10, %11) : (memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
  call @print_memref_f32(%8) : (memref<*xf32>) -> ()
  return
}

func private @mgpuMemGetDeviceMemRef1dFloat(%ptr : memref<?xf32>) -> (memref<?xf32>)
func private @print_memref_f32(%ptr : memref<*xf32>)
