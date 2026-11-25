// RUN: mlir-opt %s --gpu-attach-select-object="priority=101" | FileCheck %s
// RUN: mlir-opt %s --gpu-attach-select-object="priority=101 module=no_.*" | FileCheck %s --check-prefix=CHECK-MATCH
// RUN: mlir-opt %s --gpu-attach-select-object | FileCheck %s --check-prefix=CHECK-DEFAULT
// RUN: mlir-opt %s --gpu-attach-select-object="priority=65535" | FileCheck %s --check-prefix=CHECK-MAX
// RUN: not mlir-opt %s --gpu-attach-select-object="priority=65536" 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID

// CHECK-INVALID: Invalid priority specified: 65536 (expected a value in the range [0, 65535])

module attributes {gpu.container_module} {
  // A module with no handler selects the first object.
  // CHECK: gpu.module @no_handler<#gpu.select_object<0 : i64, priority = 101>> [#nvvm.target]
  // CHECK-MATCH: gpu.module @no_handler<#gpu.select_object<0 : i64, priority = 101>> [#nvvm.target]
  // CHECK-DEFAULT: gpu.module @no_handler [#nvvm.target]
  // CHECK-MAX: gpu.module @no_handler<#gpu.select_object<0 : i64, priority = 65535>> [#nvvm.target]
  gpu.module @no_handler [#nvvm.target] {
  }

  // The target of an existing handler is kept.
  // CHECK: gpu.module @with_target<#gpu.select_object<1 : i64, priority = 101>>
  // CHECK-MATCH: gpu.module @with_target<#gpu.select_object<1 : i64>>
  // CHECK-DEFAULT: gpu.module @with_target<#gpu.select_object<1 : i64>>
  gpu.module @with_target <#gpu.select_object<1>> [#nvvm.target, #nvvm.target<chip = "sm_90">] {
  }

  // The priority of an existing handler is kept.
  // CHECK: gpu.module @with_priority<#gpu.select_object<0 : i64, priority = 7>>
  gpu.module @with_priority <#gpu.select_object<0, priority = 7>> [#nvvm.target] {
  }
}
