; RUN: llc -O3 -march=amdgcn -verify-machineinstrs < %s | FileCheck --check-prefix=CHECK %s

; SIInsertWaitcnts should preserve waitcnt instructions coming from the user

; CHECK-LABEL: test_waitcnt_asm
; CHECK: s_waitcnt vmcnt(0)
; CHECK: s_waitcnt vmcnt(0)
; CHECK: s_waitcnt vmcnt(0)
; CHECK-NOT: s_waitcnt
; CHECK: s_endpgm
define amdgpu_kernel void @test_waitcnt_asm() {
  call void asm sideeffect "s_waitcnt vmcnt(0)", ""()
  call void asm sideeffect "s_waitcnt vmcnt(0)", ""()
  call void asm sideeffect "s_waitcnt vmcnt(0)", ""()
  ret void
}

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

; CHECK-LABEL: test_waitcnt_builtin
; CHECK: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NOT: s_waitcnt
; CHECK: s_endpgm
define amdgpu_kernel void @test_waitcnt_builtin() {
  call void @llvm.amdgcn.s.waitcnt(i32 0)
  call void @llvm.amdgcn.s.waitcnt(i32 0)
  call void @llvm.amdgcn.s.waitcnt(i32 0)
  ret void
}

; CHECK-LABEL: test_waitcnt_builtin_non_kernel
; CHECK: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NOT: s_waitcnt
; CHECK: s_setpc
define void @test_waitcnt_builtin_non_kernel() {
  call void @llvm.amdgcn.s.waitcnt(i32 0)
  call void @llvm.amdgcn.s.waitcnt(i32 0)
  call void @llvm.amdgcn.s.waitcnt(i32 0)
  ret void
}

declare void @llvm.amdgcn.s.waitcnt(i32)
