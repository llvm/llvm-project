; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -stress-regalloc=2 -filetype=null %s 2>&1 | FileCheck %s

; A negative test to capture the expected error when the VGPRs are insufficient for wwm-regalloc.

; CHECK: error: can't find enough VGPRs for wwm-regalloc

define amdgpu_kernel void @test(i32 %in) {
entry:
  call void asm sideeffect "", "~{v[0:7]}" ()
  call void asm sideeffect "", "~{v[8:15]}" ()
  call void asm sideeffect "", "~{v[16:23]}" ()
  call void asm sideeffect "", "~{v[24:31]}" ()
  call void asm sideeffect "", "~{v[32:39]}" ()
  call void asm sideeffect "", "~{v[40:47]}" ()
  call void asm sideeffect "", "~{v[48:55]}" ()
  call void asm sideeffect "", "~{v[56:63]}" ()
  %val0 = call i32 asm sideeffect "; def $0", "=s" ()
  %val1 = call i32 asm sideeffect "; def $0", "=s" ()
  %val2 = call i32 asm sideeffect "; def $0", "=s" ()
  %cmp = icmp eq i32 %in, 0
  br i1 %cmp, label %bb0, label %ret
bb0:
  call void asm sideeffect "; use $0", "s"(i32 %val0)
  call void asm sideeffect "; use $0", "s"(i32 %val1)
  call void asm sideeffect "; use $0", "s"(i32 %val2)
  br label %ret
ret:
  ret void
}
