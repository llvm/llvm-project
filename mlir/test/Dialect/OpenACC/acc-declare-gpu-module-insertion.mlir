// RUN: mlir-opt %s -split-input-file -acc-declare-gpu-module-insertion | FileCheck %s

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

// -----

// If the GPU module already has the global (e.g. from CUDA Fortran pass before
// ACCImplicitDeclare marked the host), reuse it and propagate acc.declare.
// CHECK-LABEL: module attributes {gpu.container_module}
// CHECK: memref.global @precloned {{.*}} {acc.declare = #acc.declare<dataClause = acc_copyin>}
// CHECK: gpu.module @acc_gpu_module {
// CHECK-NEXT: memref.global @precloned {{.*}} {acc.declare = #acc.declare<dataClause = acc_copyin>}
// CHECK-NEXT: }

module attributes {gpu.container_module} {
  memref.global @precloned : memref<4xf32> = dense<0.0> {acc.declare = #acc.declare<dataClause = acc_copyin>}
  gpu.module @acc_gpu_module {
    memref.global @precloned : memref<4xf32> = dense<0.0>
  }
}
