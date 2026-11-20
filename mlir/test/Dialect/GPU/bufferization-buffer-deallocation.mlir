// RUN: mlir-opt %s -buffer-deallocation-pipeline -split-input-file | FileCheck %s

func.func @gpu_launch() {
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%arg0, %arg1, %arg2) in (%arg6 = %c1, %arg7 = %c1, %arg8 = %c1)
    threads(%arg3, %arg4, %arg5) in (%arg9 = %c1, %arg10 = %c1, %arg11 = %c1) {
    %alloc = memref.alloc() : memref<2xf32>
    "test.read_buffer"(%alloc) : (memref<2xf32>) -> ()
    gpu.terminator
  }
  return
}

// CHECK-LABEL: func @gpu_launch
//       CHECK:   gpu.launch
//       CHECK:     [[ALLOC:%.+]] = memref.alloc(
//       CHECK:     memref.dealloc [[ALLOC]]
//       CHECK:     gpu.terminator

// -----

module attributes {gpu.container_module} {
  gpu.binary @kernels [#gpu.object<#nvvm.target, "">]

  func.func @gpu_launch_func() {
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<2xf32>
    gpu.launch_func @kernels::@kernel blocks in (%c1, %c1, %c1)
        threads in (%c1, %c1, %c1) args(%alloc : memref<2xf32>)
    return
  }
}

// CHECK-LABEL: func @gpu_launch_func
//       CHECK:   [[ALLOC:%.+]] = memref.alloc(
//       CHECK:   gpu.launch_func
//  CHECK-NEXT:   memref.dealloc [[ALLOC]]
