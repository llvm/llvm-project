; The emitted Tag_CSKY_FPU_ABI build attribute (17) reflects the floating-point
; ABI: hard-float ABI (from the "float-abi" module flag) emits FPU_ABI_HARD (3),
; while the soft-float ABI with hard-float instructions available emits
; FPU_ABI_SOFTFP (2), and no float unit emits FPU_ABI_SOFT (1).

; RUN: split-file %s %t

; Hard float ABI module flag with an FPU: FPU_ABI_HARD (3).
; RUN: llc -mtriple=csky -mattr=+2e3,+fpuv2_sf,+fpuv2_df,+hard-float < %t/hard.ll | FileCheck %s --check-prefix=HARD

; Soft float ABI (target default) with an FPU: FPU_ABI_SOFTFP (2).
; RUN: llc -mtriple=csky -mattr=+2e3,+fpuv2_sf,+fpuv2_df,+hard-float < %t/none.ll | FileCheck %s --check-prefix=SOFTFP

; Explicit soft float ABI module flag with an FPU: FPU_ABI_SOFTFP (2).
; RUN: llc -mtriple=csky -mattr=+2e3,+fpuv2_sf,+fpuv2_df,+hard-float < %t/soft.ll | FileCheck %s --check-prefix=SOFTFP

; No float unit: FPU_ABI_SOFT (1), regardless of the module flag.
; RUN: llc -mtriple=csky -mattr=+2e3 < %t/hard.ll | FileCheck %s --check-prefix=SOFT

; HARD: .csky_attribute 17, 3
; SOFTFP: .csky_attribute 17, 2
; SOFT: .csky_attribute 17, 1

;--- none.ll
define void @f() {
  ret void
}

;--- hard.ll
define void @f() {
  ret void
}
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"float-abi", !"hard"}

;--- soft.ll
define void @f() {
  ret void
}
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"float-abi", !"soft"}
