; RUN: not llc -mtriple=amdgpu9.00 < %s 2>&1 | FileCheck -check-prefixes=GCN %s
; RUN: not llc -mtriple=amdgpu9.06 < %s 2>&1 | FileCheck -check-prefixes=GCN %s

; GCN:     could not allocate input reg for constraint 'a'

define amdgpu_kernel void @used_1a() {
  call void asm sideeffect "", "a"(i32 1)
  ret void
}
