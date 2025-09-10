// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx942 | FileCheck %s --check-prefixes=CHECK,GFX9
// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx1030 | FileCheck %s --check-prefixes=CHECK,GFX10
// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx1100 | FileCheck %s --check-prefixes=CHECK,GFX11
// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx1201 | FileCheck %s --check-prefixes=CHECK,GFX12

// CHECK-LABEL: func @memory_counter_wait
func.func @memory_counter_wait() {
  // GFX9: rocdl.s.waitcnt 53119
  // GFX10: rocdl.s.waitcnt 65407
  // GFX11: rocdl.s.waitcnt 65527
  // GFX12-NOT: rocdl.s.wait.loadcnt
  // GFX12-NOT: rocdl.s.wait.storecnt
  // GFX12-NOT: rocdl.s.wait.expcnt
  // GFX12-NOT: rocdl.s.wait.dscnt
  amdgpu.memory_counter_wait

  // GFX9: rocdl.s.waitcnt 3952
  // GFX10: rocdl.s.waitcnt 16240
  // GFX11: rocdl.s.waitcnt 1015
  // GFX12: rocdl.s.wait.loadcnt 0
  amdgpu.memory_counter_wait load(0)

  // GFX9: rocdl.s.waitcnt 3952
  // GFX10: rocdl.s.waitcnt 16240
  // GFX11: rocdl.s.waitcnt 1015
  // GFX12: rocdl.s.wait.storecnt 0
  amdgpu.memory_counter_wait store(0)

  // GFX9: rocdl.s.waitcnt 53007
  // GFX10: rocdl.s.waitcnt 65295
  // GFX11: rocdl.s.waitcnt 65520
  // GFX12: rocdl.s.wait.expcnt 0
  amdgpu.memory_counter_wait exp(0)

  // GFX9: rocdl.s.waitcnt 49279
  // GFX10: rocdl.s.waitcnt 49279
  // GFX11: rocdl.s.waitcnt 64519
  // GFX12: rocdl.s.wait.dscnt 0
  amdgpu.memory_counter_wait ds(0)

  return
}
