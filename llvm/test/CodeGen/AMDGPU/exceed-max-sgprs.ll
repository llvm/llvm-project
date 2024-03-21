; RUN: not llc -mtriple=amdgcn -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=ERROR %s

; ERROR: error: <unknown>:0:0: scalar registers (106) exceeds limit (104) in function 'use_too_many_sgprs_tahiti'
define amdgpu_kernel void @use_too_many_sgprs_tahiti() nounwind "target-cpu"="tahiti" {
  call void asm sideeffect "", "~{s[0:7]}" ()
  call void asm sideeffect "", "~{s[8:15]}" ()
  call void asm sideeffect "", "~{s[16:23]}" ()
  call void asm sideeffect "", "~{s[24:31]}" ()
  call void asm sideeffect "", "~{s[32:39]}" ()
  call void asm sideeffect "", "~{s[40:47]}" ()
  call void asm sideeffect "", "~{s[48:55]}" ()
  call void asm sideeffect "", "~{s[56:63]}" ()
  call void asm sideeffect "", "~{s[64:71]}" ()
  call void asm sideeffect "", "~{s[72:79]}" ()
  call void asm sideeffect "", "~{s[80:87]}" ()
  call void asm sideeffect "", "~{s[88:95]}" ()
  call void asm sideeffect "", "~{s[96:103]}" ()
  call void asm sideeffect "", "~{vcc}" ()
  ret void
}

; ERROR: error: <unknown>:0:0: scalar registers (106) exceeds limit (104) in function 'use_too_many_sgprs_bonaire'
define amdgpu_kernel void @use_too_many_sgprs_bonaire() nounwind "target-cpu"="bonaire" {
  call void asm sideeffect "", "~{s[0:7]}" ()
  call void asm sideeffect "", "~{s[8:15]}" ()
  call void asm sideeffect "", "~{s[16:23]}" ()
  call void asm sideeffect "", "~{s[24:31]}" ()
  call void asm sideeffect "", "~{s[32:39]}" ()
  call void asm sideeffect "", "~{s[40:47]}" ()
  call void asm sideeffect "", "~{s[48:55]}" ()
  call void asm sideeffect "", "~{s[56:63]}" ()
  call void asm sideeffect "", "~{s[64:71]}" ()
  call void asm sideeffect "", "~{s[72:79]}" ()
  call void asm sideeffect "", "~{s[80:87]}" ()
  call void asm sideeffect "", "~{s[88:95]}" ()
  call void asm sideeffect "", "~{s[96:103]}" ()
  call void asm sideeffect "", "~{vcc}" ()
  ret void
}

; ERROR: error: <unknown>:0:0: scalar registers (108) exceeds limit (104) in function 'use_too_many_sgprs_bonaire_flat_scr'
define amdgpu_kernel void @use_too_many_sgprs_bonaire_flat_scr() nounwind "target-cpu"="bonaire" {
  call void asm sideeffect "", "~{s[0:7]}" ()
  call void asm sideeffect "", "~{s[8:15]}" ()
  call void asm sideeffect "", "~{s[16:23]}" ()
  call void asm sideeffect "", "~{s[24:31]}" ()
  call void asm sideeffect "", "~{s[32:39]}" ()
  call void asm sideeffect "", "~{s[40:47]}" ()
  call void asm sideeffect "", "~{s[48:55]}" ()
  call void asm sideeffect "", "~{s[56:63]}" ()
  call void asm sideeffect "", "~{s[64:71]}" ()
  call void asm sideeffect "", "~{s[72:79]}" ()
  call void asm sideeffect "", "~{s[80:87]}" ()
  call void asm sideeffect "", "~{s[88:95]}" ()
  call void asm sideeffect "", "~{s[96:103]}" ()
  call void asm sideeffect "", "~{vcc}" ()
  call void asm sideeffect "", "~{flat_scratch}" ()
  ret void
}

; ERROR: error: <unknown>:0:0: scalar registers (98) exceeds limit (96) in function 'use_too_many_sgprs_iceland'
define amdgpu_kernel void @use_too_many_sgprs_iceland() nounwind "target-cpu"="iceland" {
  call void asm sideeffect "", "~{vcc}" ()
  call void asm sideeffect "", "~{s[0:7]}" ()
  call void asm sideeffect "", "~{s[8:15]}" ()
  call void asm sideeffect "", "~{s[16:23]}" ()
  call void asm sideeffect "", "~{s[24:31]}" ()
  call void asm sideeffect "", "~{s[32:39]}" ()
  call void asm sideeffect "", "~{s[40:47]}" ()
  call void asm sideeffect "", "~{s[48:55]}" ()
  call void asm sideeffect "", "~{s[56:63]}" ()
  call void asm sideeffect "", "~{s[64:71]}" ()
  call void asm sideeffect "", "~{s[72:79]}" ()
  call void asm sideeffect "", "~{s[80:87]}" ()
  call void asm sideeffect "", "~{s[88:95]}" ()
  ret void
}

; ERROR: error: <unknown>:0:0: addressable scalar registers (103) exceeds limit (102) in function 'use_too_many_sgprs_fiji'
define amdgpu_kernel void @use_too_many_sgprs_fiji() nounwind "target-cpu"="fiji" {
  call void asm sideeffect "", "~{s[0:7]}" ()
  call void asm sideeffect "", "~{s[8:15]}" ()
  call void asm sideeffect "", "~{s[16:23]}" ()
  call void asm sideeffect "", "~{s[24:31]}" ()
  call void asm sideeffect "", "~{s[32:39]}" ()
  call void asm sideeffect "", "~{s[40:47]}" ()
  call void asm sideeffect "", "~{s[48:55]}" ()
  call void asm sideeffect "", "~{s[56:63]}" ()
  call void asm sideeffect "", "~{s[64:71]}" ()
  call void asm sideeffect "", "~{s[72:79]}" ()
  call void asm sideeffect "", "~{s[80:87]}" ()
  call void asm sideeffect "", "~{s[88:95]}" ()
  call void asm sideeffect "", "~{s[96:99]}" ()
  call void asm sideeffect "", "~{s[100:101]}" ()
  call void asm sideeffect "", "~{s102}" ()
  ret void
}
