// RUN: mlir-opt -gpu-async-region %s | FileCheck %s

// CHECK: module attributes {gpu.container_module}
module attributes {gpu.container_module} {

  gpu.module @kernels {
    gpu.func @kernel() kernel { gpu.return }
  }

  func private @foo() -> ()

  // CHECK-LABEL:func @async(%{{.*}}: index)
  func @async(%sz : index) {
    // CHECK: %[[t0:.*]] = gpu.wait async
    // CHECK: %[[t1:.*]] = gpu.launch_func async [%[[t0]]]
    gpu.launch_func @kernels::@kernel
        blocks in (%sz, %sz, %sz) threads in (%sz, %sz, %sz)
    // CHECK: %[[t2:.*]] = gpu.launch_func async [%[[t1]]]
    gpu.launch_func @kernels::@kernel
        blocks in (%sz, %sz, %sz) threads in (%sz, %sz, %sz)
    // CHECK: gpu.wait [%[[t2]]]
    // CHECK: call @foo
    call @foo() : () -> ()
    return
  }

}
