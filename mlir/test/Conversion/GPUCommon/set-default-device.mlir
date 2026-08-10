// RUN: mlir-opt %s --gpu-to-llvm | FileCheck %s

module attributes {gpu.container_module} {
  // CHECK-LABEL: func @set_default_device
  func.func @set_default_device(%arg0: i32) {
    // CHECK: mgpuSetDefaultDevice
    gpu.set_default_device %arg0
    return
  }
}
