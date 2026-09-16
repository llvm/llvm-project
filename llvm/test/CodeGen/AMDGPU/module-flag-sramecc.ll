; Test that sramecc settings are controlled by the amdgpu.sramecc
; module flag.

; RUN: split-file %s %t
; RUN: llc -mtriple=amdgpu9.06-amd-amdhsa < %t/on.ll | FileCheck --check-prefix=SRAMECC-ON %s
; RUN: llc -mtriple=amdgpu9.06-amd-amdhsa < %t/off.ll | FileCheck --check-prefix=SRAMECC-OFF %s
; RUN: llc -mtriple=amdgpu9.06-amd-amdhsa < %t/absent.ll | FileCheck --check-prefix=SRAMECC-ANY %s

; Test that the is ignored on targets that don't support it. gfx906 supports sramecc, gfx900 does not.
; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa < %t/on.ll | FileCheck --check-prefix=GFX900 %s

; Target directives for supported target
; SRAMECC-ON: .amdgcn_target "amdgpu9.06-amd-amdhsa-unknown-gfx906:sramecc+"
; SRAMECC-OFF: .amdgcn_target "amdgpu9.06-amd-amdhsa-unknown-gfx906:sramecc-"
; SRAMECC-ANY: .amdgcn_target "amdgpu9.06-amd-amdhsa-unknown-gfx906"

; Unsupported target ignores the flag
; GFX900: .amdgcn_target "amdgpu9.00-amd-amdhsa-unknown-gfx900"

; When sramecc is on, avoid _d16_hi
; SRAMECC-ON-LABEL: {{^}}load_d16:
; SRAMECC-ON: s_waitcnt
; SRAMECC-ON: flat_load_ushort v{{[0-9]+}}, v[{{[0-9:]+}}]
; SRAMECC-ON: s_setpc_b64

; When sramecc is off, use _d16_hi instructions
; SRAMECC-OFF-LABEL: {{^}}load_d16:
; SRAMECC-OFF: s_waitcnt
; SRAMECC-OFF: flat_load_short_d16_hi v0, v[{{[0-9:]+}}]
; SRAMECC-OFF: s_setpc_b64

; SRAMECC-ANY-LABEL: {{^}}load_d16:
; SRAMECC-ANY: s_waitcnt
; SRAMECC-ANY: flat_load_ushort v{{[0-9]+}}, v[{{[0-9:]+}}]
; SRAMECC-ANY: s_setpc_b64

; Unsupported target (gfx900) ignores sramecc flag and uses _d16_hi
; GFX900-LABEL: {{^}}load_d16:
; GFX900: s_waitcnt
; GFX900: flat_load_short_d16_hi v0, v[{{[0-9:]+}}]
; GFX900: s_setpc_b64

;--- on.ll
define <2 x i16> @load_d16(<2 x i16> %vec, ptr %ptr) {
  %val = load i16, ptr %ptr
  %result = insertelement <2 x i16> %vec, i16 %val, i32 1
  ret <2 x i16> %result
}
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu.sramecc", i32 1}

;--- off.ll
define <2 x i16> @load_d16(<2 x i16> %vec, ptr %ptr) {
  %val = load i16, ptr %ptr
  %result = insertelement <2 x i16> %vec, i16 %val, i32 1
  ret <2 x i16> %result
}
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu.sramecc", i32 0}

;--- absent.ll
define <2 x i16> @load_d16(<2 x i16> %vec, ptr %ptr) {
  %val = load i16, ptr %ptr
  %result = insertelement <2 x i16> %vec, i16 %val, i32 1
  ret <2 x i16> %result
}
