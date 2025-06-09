; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs=0 < %s | FileCheck -enable-var-scope %s

declare hidden void @external_void_func_a15i32_inreg([15 x i32] inreg) #0
declare hidden void @external_void_func_a16i32_inreg([16 x i32] inreg) #0
declare hidden void @external_void_func_a15i32_inreg_i32_inreg([15 x i32] inreg, i32 inreg) #0

define void @test_call_external_void_func_a15i32_inreg([15 x i32] inreg %arg0) #0 {
  call void @external_void_func_a15i32_inreg([15 x i32] inreg %arg0)
  ret void
}

define void @test_call_external_void_func_a16i32_inreg([16 x i32] inreg %arg0) #0 {
  call void @external_void_func_a16i32_inreg([16 x i32] inreg %arg0)
  ret void
}

define void @test_call_external_void_func_a15i32_inreg_i32_inreg([15 x i32] inreg %arg0, i32 inreg %arg1) #0 {
  call void @external_void_func_a15i32_inreg_i32_inreg([15 x i32] inreg %arg0, i32 inreg %arg1)
  ret void
}

attributes #0 = { nounwind }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}
; CHECK: v_readlane_b32
; CHECK: s_mov_b32
; CHECK: v_writelane_b32
; CHECK: s_swappc_b64
; CHECK: s_or_saveexec_b64
; CHECK: buffer_load_dword
; CHECK: s_waitcnt
; CHECK: s_addk_i32
; CHECK: v_readfirstlane_b32
; CHECK: s_mov_b64