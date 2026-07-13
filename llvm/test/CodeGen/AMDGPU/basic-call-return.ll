; RUN: llc -mtriple=amdgpu8.03-amd-amdhsa < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgpu7.01-amd-amdhsa < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa < %s | FileCheck -check-prefix=GCN %s

define void @void_func_void() #2 {
  ret void
}

; GCN-LABEL: {{^}}test_call_void_func_void:
define amdgpu_kernel void @test_call_void_func_void() {
  call void @void_func_void()
  ret void
}

define void @void_func_void_clobber_s40_s41() #2 {
  call void asm sideeffect "", "~{s[40:41]}"() #0
  ret void
}

define amdgpu_kernel void @test_call_void_func_void_clobber_s40_s41() {
  call void @void_func_void_clobber_s40_s41()
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind noinline }
