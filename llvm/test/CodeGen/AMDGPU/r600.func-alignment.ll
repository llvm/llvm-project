; RUN: llc < %s -mtriple=r600 -mcpu=redwood | FileCheck %s
; RUN: llc < %s -mtriple=r600 -mcpu=rs880 | FileCheck  %s

; CHECK: .globl test
; Functions need to be cacheline (256B) aligned to prevent GPU hangs
; CHECK: .p2align 8
; CHECK: {{^}}test:
; CHECK: CF_END

define amdgpu_ps void @test(<4 x float> inreg %reg0) {
entry:
  ret void
}
