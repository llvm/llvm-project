// RUN: mlir-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=lane" | FileCheck %s

// `convert-math-to-xevm` only maps the Math ops that have a native or OpenCL
// counterpart. The rest must still be lowered, or they survive into
// serialization and translation to LLVM IR fails.
// See https://github.com/llvm/llvm-project/issues/213583.

// CHECK-LABEL: gpu.module @kernels
// CHECK-NOT: math.
module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @classify_inf(%input : memref<1xf32>,
                           %output : memref<1xi32>) kernel {
      %c0 = arith.constant 0 : index
      %zero = arith.constant 0 : i32
      %one = arith.constant 1 : i32
      %value = memref.load %input[%c0] : memref<1xf32>
      %is_inf = math.isinf %value : f32
      %result = arith.select %is_inf, %one, %zero : i32
      memref.store %result, %output[%c0] : memref<1xi32>
      gpu.return
    }
  }
}
