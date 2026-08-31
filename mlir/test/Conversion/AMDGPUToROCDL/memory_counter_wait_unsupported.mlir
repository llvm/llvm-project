// RUN: mlir-opt %s --verify-diagnostics --convert-amdgpu-to-rocdl=triple=amdgpu9.42-amd-amdhsa
// RUN: mlir-opt %s --verify-diagnostics --convert-amdgpu-to-rocdl=triple=amdgpu10.30-amd-amdhsa
// RUN: mlir-opt %s --verify-diagnostics --convert-amdgpu-to-rocdl=triple=amdgpu11.00-amd-amdhsa

func.func @memory_counter_wait_tensor() {
  // expected-error @below{{failed to legalize operation 'amdgpu.memory_counter_wait'}}
  // expected-error @below{{'amdgpu.memory_counter_wait' op unsupported chipset}}
  amdgpu.memory_counter_wait tensor(0)

  return
}
