// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx942 | FileCheck %s --check-prefixes=CHECK,GFX9
// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx1030 | FileCheck %s --check-prefixes=CHECK,GFX10
// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx1100 | FileCheck %s --check-prefixes=CHECK,GFX11


// CHECK-LABEL: func @waitcnt
func.func @waitcnt() {
  // GFX9: rocdl.s.waitcnt 53119
  // GFX10: rocdl.s.waitcnt 65407
  // GFX11: rocdl.s.waitcnt 65527
  amdgpu.waitcnt

  // GFX9: rocdl.s.waitcnt 3952
  // GFX10: rocdl.s.waitcnt 16240
  // GFX11: rocdl.s.waitcnt 1015
  amdgpu.waitcnt vmcnt(0)

  // GFX9: rocdl.s.waitcnt 53007
  // GFX10: rocdl.s.waitcnt 65295
  // GFX11: rocdl.s.waitcnt 65520
  amdgpu.waitcnt expcnt(0)

  // GFX9: rocdl.s.waitcnt 49279
  // GFX10: rocdl.s.waitcnt 49279
  // GFX11: rocdl.s.waitcnt 64519
  amdgpu.waitcnt lgkmcnt(0)

  return
}
