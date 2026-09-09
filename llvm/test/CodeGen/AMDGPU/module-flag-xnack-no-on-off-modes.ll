; Test targets without xnack on/off mode support ignore module flags
; Targets with only FEATURE_XNACK (but not FEATURE_XNACK_ON_OFF_MODES)
; have xnack always on and ignore module flag settings.
; The target ID should not contain the xnack specifier.

; RUN: split-file %s %t
; RUN: llc -mtriple=amdgpu12.5-amd-amdhsa < %t/on.ll | FileCheck --check-prefix=CHECK %s
; RUN: llc -mtriple=amdgpu12.5-amd-amdhsa < %t/off.ll | FileCheck --check-prefix=CHECK %s
; RUN: llc -mtriple=amdgpu12.5-amd-amdhsa < %t/absent.ll | FileCheck --check-prefix=CHECK %s

; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa < %t/on.ll | FileCheck --check-prefix=CHECK %s
; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa < %t/off.ll | FileCheck --check-prefix=CHECK %s
; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa < %t/absent.ll | FileCheck --check-prefix=CHECK %s

; RUN: llc -mtriple=amdgpu12.51-amd-amdhsa < %t/on.ll | FileCheck --check-prefix=CHECK %s
; RUN: llc -mtriple=amdgpu12.51-amd-amdhsa < %t/off.ll | FileCheck --check-prefix=CHECK %s
; RUN: llc -mtriple=amdgpu12.51-amd-amdhsa < %t/absent.ll | FileCheck --check-prefix=CHECK %s

; Module flags are ignored - target ID has no xnack specifier
; CHECK: .amdgcn_target "amdgpu12.5{{[0-1]?}}-amd-amdhsa-unknown-gfx{{12-5-generic|1250|1251}}"

; Targets without on/off mode support always have xnack enabled,
; so loads must not overwrite pointer arguments (same behavior regardless of module flag)
; CHECK-LABEL: {{^}}simple_clause:
; CHECK: flat_load_b32 v4, v[0:1]
; CHECK-NEXT: flat_load_b32 v5, v[2:3]

;--- on.ll
define i32 @simple_clause(ptr %ptr0, ptr %ptr1) {
  %val0 = load i32, ptr %ptr0
  %val1 = load i32, ptr %ptr1
  %add = add i32 %val0, %val1
  ret i32 %add
}
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu.xnack", i32 1}

;--- off.ll
define i32 @simple_clause(ptr %ptr0, ptr %ptr1) {
  %val0 = load i32, ptr %ptr0
  %val1 = load i32, ptr %ptr1
  %add = add i32 %val0, %val1
  ret i32 %add
}
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu.xnack", i32 0}

;--- absent.ll
define i32 @simple_clause(ptr %ptr0, ptr %ptr1) {
  %val0 = load i32, ptr %ptr0
  %val1 = load i32, ptr %ptr1
  %add = add i32 %val0, %val1
  ret i32 %add
}
