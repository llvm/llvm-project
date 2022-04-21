; RUN: not llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefixes=GCN %s
; RUN: not llc -march=amdgcn -mcpu=gfx906 -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefixes=GCN %s

; GCN:     couldn't allocate input reg for constraint 'a'

define amdgpu_kernel void @used_1a() {
  call void asm sideeffect "", "a"(i32 1)
  ret void
}
