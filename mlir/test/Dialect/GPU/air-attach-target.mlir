// RUN: mlir-opt %s --air-attach-target='module=air.* chip=agx-g14c features=+agx-strict O=2' | FileCheck %s
// RUN: mlir-opt %s --air-attach-target='module=air_metal.*' | FileCheck %s --check-prefix=METAL

module attributes {gpu.container_module} {
  // CHECK: @air_module_1 [#air.target<O = 2, chip = "agx-g14c", features = "+agx-strict">]
  // METAL: @air_module_1 {
  gpu.module @air_module_1 {
  }
  // CHECK: @air_metal [#air.target<O = 2, chip = "agx-g14c", features = "+agx-strict">]
  // METAL: @air_metal [#air.target]
  gpu.module @air_metal {
  }
  // CHECK: @other {
  gpu.module @other {
  }
}
