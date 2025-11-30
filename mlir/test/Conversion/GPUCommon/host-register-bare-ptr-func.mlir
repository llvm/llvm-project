// RUN: mlir-opt %s --gpu-to-llvm="use-bare-pointers-for-host=1" -split-input-file -verify-diagnostics

module attributes {gpu.container_module} {
  func.func @dynamic(%buf : memref<?xf32>) {
    // expected-error @+1 {{cannot lower memref with bare pointer calling convention}}
    gpu.host_register %buf : memref<?xf32>
    return
  }
}
