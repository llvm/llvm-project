; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=hawaii -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

define void @void_func_void() #2 {
  ret void
}

; GCN-LABEL: {{^}}test_call_void_func_void:
define amdgpu_kernel void @test_call_void_func_void() {
  call void @void_func_void()
  ret void
}

define void @void_func_void_clobber_s40_s41() #2 {
  call void asm sideeffect "", "~{SGPR40_SGPR41}"() #0
  ret void
}

define amdgpu_kernel void @test_call_void_func_void_clobber_s40_s41() {
  call void @void_func_void_clobber_s40_s41()
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind noinline }
