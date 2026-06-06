// RUN: mlir-opt %s --verify-diagnostics --convert-amdgpu-to-rocdl=chipset=gfx942
// RUN: mlir-opt %s --verify-diagnostics --convert-amdgpu-to-rocdl=chipset=gfx1030
// RUN: mlir-opt %s --verify-diagnostics --convert-amdgpu-to-rocdl=chipset=gfx1100

func.func @memory_counter_wait_tensor() {
  // expected-error @below{{failed to legalize operation 'amdgpu.memory_counter_wait'}}
  // expected-error @below{{'amdgpu.memory_counter_wait' op unsupported chipset}}
  amdgpu.memory_counter_wait tensor(0)

  return
}
