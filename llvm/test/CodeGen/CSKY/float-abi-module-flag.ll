; The "float-abi" module flag selects the floating-point calling convention.
; RUN: split-file %s %t

; Soft float ABI (the target default): FP arguments are passed in GPRs and moved
; into VFP registers with fmtvrl.
; RUN: llc -csky-no-aliases -mtriple=csky -mattr=+2e3,+fpuv2_sf,+fpuv2_df,+hard-float < %t/none.ll | FileCheck %s --check-prefix=SOFT

; Hard float ABI module flag: FP arguments arrive directly in VFP registers, so
; no fmtvrl moves are needed.
; RUN: llc -csky-no-aliases -mtriple=csky -mattr=+2e3,+fpuv2_sf,+fpuv2_df,+hard-float < %t/hard.ll | FileCheck %s --check-prefix=HARD

; Soft float ABI module flag matches the default.
; RUN: llc -csky-no-aliases -mtriple=csky -mattr=+2e3,+fpuv2_sf,+fpuv2_df,+hard-float < %t/soft.ll | FileCheck %s --check-prefix=SOFT

; The legacy -float-abi target option still selects the ABI when the module has
; no "float-abi" flag.
; RUN: llc -csky-no-aliases -mtriple=csky -mattr=+2e3,+fpuv2_sf,+fpuv2_df,+hard-float -float-abi=hard < %t/none.ll | FileCheck %s --check-prefix=HARD

; An explicit module flag takes precedence over the legacy -float-abi option.
; RUN: llc -csky-no-aliases -mtriple=csky -mattr=+2e3,+fpuv2_sf,+fpuv2_df,+hard-float -float-abi=hard < %t/soft.ll | FileCheck %s --check-prefix=SOFT

;--- none.ll
define float @f(float %x, float %y) {
  %r = fadd float %x, %y
  ret float %r
}

;--- hard.ll
define float @f(float %x, float %y) {
  %r = fadd float %x, %y
  ret float %r
}
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"float-abi", !"hard"}

;--- soft.ll
define float @f(float %x, float %y) {
  %r = fadd float %x, %y
  ret float %r
}
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"float-abi", !"soft"}

; SOFT: fmtvrl
; HARD-NOT: fmtvrl
