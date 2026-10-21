// RUN: mlir-opt %s --agxir-attach-target='module=agx.* chip=agx-g13c O=3' | FileCheck %s
// RUN: mlir-opt %s --agxir-attach-target='chip=agx-g14c features=+mesa-isa' | FileCheck %s --check-prefix=FEAT

module attributes {gpu.container_module} {
  // CHECK: @agx_module [#agxir.target<O = 3, chip = "agx-g13c">]
  gpu.module @agx_module {
  }
  // CHECK: @other_module {
  gpu.module @other_module {
  }
}
// FEAT: #agxir.target<chip = "agx-g14c", features = "+mesa-isa">
