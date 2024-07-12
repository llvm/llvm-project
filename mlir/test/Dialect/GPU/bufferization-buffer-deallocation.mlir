// RUN: mlir-opt %s -buffer-deallocation-pipeline --allow-unregistered-dialect | FileCheck %s

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
