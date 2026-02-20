// RUN: mlir-opt %s -acc-declare-gpu-module-insertion | FileCheck %s

// Test that globals with acc.declare are copied into the GPU module.
// The host global stays in the module; a copy is inserted into the GPU module.

// CHECK-LABEL: module
// CHECK: memref.global @arr {{.*}} {acc.declare = #acc.declare<dataClause = acc_create>}
// CHECK: gpu.module @acc_gpu_module {
// CHECK: memref.global @arr {{.*}} {acc.declare = #acc.declare<dataClause = acc_create>}
// CHECK: }

module {
  memref.global @arr : memref<7xf32> = dense<0.0> {acc.declare = #acc.declare<dataClause = acc_create>}
}
