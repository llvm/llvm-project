// RUN: mlir-opt %s --convert-amdgpu-to-rocdl=chipset=gfx1250 | FileCheck %s

// CHECK-LABEL: func @memory_counter_wait_tensor
func.func @memory_counter_wait_tensor() {
  // CHECK: rocdl.s.wait.tensorcnt 3
  amdgpu.memory_counter_wait tensor(3)

  return
}
