; RUN: llc -O3 -mtriple=amdgpu10.10 < %s | FileCheck --check-prefix=CHECK %s

; SIInsertWaitcnts should preserve waitcnt instructions coming from the user.
; s_waitcnt_vscnt only exists on gfx10+.

; CHECK-LABEL: test_waitcnt_vscnt_asm
; CHECK: s_waitcnt_vscnt null, 0x0
; CHECK: s_waitcnt_vscnt null, 0x0
; CHECK: s_waitcnt_vscnt null, 0x0
; CHECK-NOT: s_waitcnt
; CHECK: s_endpgm
define amdgpu_kernel void @test_waitcnt_vscnt_asm() {
  call void asm sideeffect "s_waitcnt_vscnt null, 0x0", ""()
  call void asm sideeffect "s_waitcnt_vscnt null, 0x0", ""()
  call void asm sideeffect "s_waitcnt_vscnt null, 0x0", ""()
  ret void
}
