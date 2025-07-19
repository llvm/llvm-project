// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx942 | FileCheck %s --check-prefixes=CHECK,GFX9
// TODO: Add more chipsets support


// CHECK-LABEL: func @waitcnt
func.func @waitcnt() {
  // GFX9: rocdl.s.waitcnt 53119
  amdgpu.waitcnt

  // GFX9: rocdl.s.waitcnt 3952
  amdgpu.waitcnt vmcnt(0)

  // GFX9: rocdl.s.waitcnt 53007
  amdgpu.waitcnt expcnt(0)

  // GFX9: rocdl.s.waitcnt 49279
  amdgpu.waitcnt lgkmcnt(0)

  return
}
