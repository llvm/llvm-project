// RUN: mlir-opt %s --nvvm-attach-target="chip=" --verify-diagnostics

// expected-error @+1 {{The target chip cannot be empty}}
module attributes {gpu.container_module} {
  gpu.module @kernel_module {
  }
}
