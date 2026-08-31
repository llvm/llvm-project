// RUN: mlir-opt %s --convert-amdgpu-to-rocdl=triple=amdgpu12.50-amd-amdhsa | FileCheck %s

// CHECK-LABEL: func @memory_counter_wait_tensor
func.func @memory_counter_wait_tensor() {
  // CHECK: rocdl.s.wait.tensorcnt 3
  amdgpu.memory_counter_wait tensor(3)

  return
}
