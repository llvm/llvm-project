; RUN: split-file %s %t

; RUN: not llc -mtriple=amdgpu6.00 -filetype=null %t/gfx6_gfx7.ll 2>&1 | FileCheck -check-prefix=GFX6_GFX7 -implicit-check-not=error %s
; RUN: not llc -mtriple=amdgpu7.04 -filetype=null %t/gfx6_gfx7.ll 2>&1 | FileCheck -check-prefix=GFX6_GFX7 -implicit-check-not=error %s
; RUN: not llc -mtriple=amdgpu8.02 -filetype=null %t/iceland.ll 2>&1 | FileCheck -check-prefix=ICELAND -implicit-check-not=error %s
; RUN: not llc -mtriple=amdgpu8.03 -filetype=null %t/fiji.ll 2>&1 | FileCheck -check-prefix=FIJI -implicit-check-not=error %s

; FIXME: Each diagnostic is emitted twice: once when the register
; limit is checked and again as the error propagates out of codegen.

; GFX6_GFX7: error: <unknown>:0:0: scalar registers (106) exceeds limit (104) in function 'use_too_many_sgprs_gfx6'
; GFX6_GFX7: error: <unknown>:0:0: scalar registers (108) exceeds limit (104) in function 'use_too_many_sgprs_bonaire_flat_scr'
; GFX6_GFX7: error: <unknown>:0:0: scalar registers (106) exceeds limit (104) in function 'use_too_many_sgprs_gfx6'
; GFX6_GFX7: error: <unknown>:0:0: scalar registers (108) exceeds limit (104) in function 'use_too_many_sgprs_bonaire_flat_scr'

; ICELAND: error: <unknown>:0:0: scalar registers (98) exceeds limit (96) in function 'use_too_many_sgprs_iceland'
; ICELAND: error: <unknown>:0:0: scalar registers (98) exceeds limit (96) in function 'use_too_many_sgprs_iceland'

; FIJI: error: <unknown>:0:0: addressable scalar registers (103) exceeds limit (102) in function 'use_too_many_sgprs_fiji'
; FIJI: error: <unknown>:0:0: addressable scalar registers (103) exceeds limit (102) in function 'use_too_many_sgprs_fiji'

;--- gfx6_gfx7.ll
define amdgpu_kernel void @use_too_many_sgprs_gfx6() {
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

define amdgpu_kernel void @use_too_many_sgprs_bonaire_flat_scr() {
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

;--- iceland.ll
define amdgpu_kernel void @use_too_many_sgprs_iceland() {
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

;--- fiji.ll
define amdgpu_kernel void @use_too_many_sgprs_fiji() {
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
