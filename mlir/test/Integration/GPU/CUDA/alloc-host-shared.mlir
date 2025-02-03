// RUN: mlir-opt %s \
// RUN: | mlir-opt -gpu-lower-to-nvvm-pipeline="cubin-format=%gpu_compilation_format" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_cuda_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// CHECK: 2000
module attributes {gpu.container_module} {
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1000_i32 = arith.constant 1000 : i32
    %memref = gpu.alloc  host_shared () : memref<1xi32>
    memref.store %c1000_i32, %memref[%c1] : memref<1xi32>
    gpu.launch blocks(%arg0, %arg1, %arg2) in (%arg6 = %c1, %arg7 = %c1, %arg8 = %c1) threads(%arg3, %arg4, %arg5) in (%arg9 = %c1, %arg10 = %c1, %arg11 = %c1) {
      %1 = memref.load %memref[%c1] : memref<1xi32>
      %2 = arith.addi %1, %1 : i32
      memref.store %2, %memref[%c1] : memref<1xi32>
      gpu.terminator
    }
    %0 = memref.load %memref[%c1] : memref<1xi32>
    vector.print %0 : i32
    return
  }
}
