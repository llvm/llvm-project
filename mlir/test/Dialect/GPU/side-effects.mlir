// RUN: mlir-opt %s --test-side-effects --verify-diagnostics

module attributes {gpu.container_module} {
  gpu.binary @kernels [#gpu.object<#nvvm.target, "">]

  func.func @launch_func(%n: index, %buffer: memref<2xf32>) {
    // expected-remark @below {{found an instance of 'read' on resource '<Default>'}}
    // expected-remark @below {{found an instance of 'write' on resource '<Default>'}}
    // expected-remark @below {{found an instance of 'allocate' on resource 'AutomaticAllocationScope'}}
    gpu.launch_func @kernels::@kernel blocks in (%n, %n, %n)
        threads in (%n, %n, %n) args(%buffer : memref<2xf32>)
    return
  }
}
