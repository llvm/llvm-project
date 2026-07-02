; Test that .amdgcn_target directive includes xnack modifier based on module flag
; Tests xnack+ (on), xnack- (off), and absent (Any) cases
; Also tests that unsupported targets ignore the xnack module flag

; RUN: split-file %s %t
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %t/on.ll | FileCheck --check-prefix=XNACK-ON %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %t/off.ll | FileCheck --check-prefix=XNACK-OFF %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %t/absent.ll | FileCheck --check-prefix=XNACK-ANY %s

; Test that xnack module flag is ignored on targets that don't support it. gfx801 supports xnack, gfx803 does not.
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx801 < %t/on.ll | FileCheck --check-prefixes=CHECK,GFX801 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 < %t/on.ll | FileCheck --check-prefixes=CHECK,GFX803 %s

; Target directives for xnack supported target
; XNACK-ON: .amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx900:xnack+"
; XNACK-OFF: .amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx900:xnack-"
; XNACK-ANY: .amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx900"

; GFX801: .amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx801:xnack+"
; GFX803: .amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx803"

; Check codegen impact - xnack affects register allocation
; When xnack is on, first load must not overwrite the pointer argument
; XNACK-ON-LABEL: {{^}}simple_clause:
; XNACK-ON: flat_load_dword v4, v[0:1]
; XNACK-ON-NEXT: flat_load_dword v5, v[2:3]

; When xnack is off, first load can overwrite the pointer argument
; XNACK-OFF-LABEL: {{^}}simple_clause:
; XNACK-OFF: flat_load_dword v0, v[0:1]
; XNACK-OFF-NEXT: flat_load_dword v1, v[2:3]

; When xnack is not specified (Any), behavior is conservative (like on)
; XNACK-ANY-LABEL: {{^}}simple_clause:
; XNACK-ANY: flat_load_dword v4, v[0:1]
; XNACK-ANY-NEXT: flat_load_dword v5, v[2:3]

; Codegen for supported vs unsupported targets
; CHECK-LABEL: {{^}}simple_clause:

; First load must not overwrite the pointer argument on gfx801 (xnack supported)
; GFX801: flat_load_dword v4, v[0:1]
; GFX801-NEXT: flat_load_dword v5, v[2:3]

; First load overwrites the pointer argument on gfx803 (xnack not supported)
; GFX803: flat_load_dword v0, v[0:1]
; GFX803-NEXT: flat_load_dword v1, v[2:3]

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
