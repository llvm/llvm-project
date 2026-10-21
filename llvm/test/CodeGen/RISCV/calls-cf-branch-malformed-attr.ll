; RUN: split-file %s %t
; RUN: not --crash llc -mtriple=riscv32 -verify-machineinstrs < %t/nonnumeric.ll 2>&1 | FileCheck %s --check-prefix=MALFORMED
; RUN: not --crash llc -mtriple=riscv32 -verify-machineinstrs < %t/negative.ll 2>&1 | FileCheck %s --check-prefix=MALFORMED
; RUN: not --crash llc -mtriple=riscv32 -verify-machineinstrs < %t/out-of-range.ll 2>&1 | FileCheck %s --check-prefix=WIDTH

;; Exercises RISCVMachineFunctionInfo::generateRawFnEntryLabel's parsing
;; of the cfi_type_hash attribute string.

; MALFORMED: LLVM ERROR: malformed cfi_type_hash attribute
; WIDTH: LLVM ERROR: landing pad Label needs to fit in unsigned 20-bits

;--- nonnumeric.ll
define void @f() "cfi_type_hash"="abc" {
  ret void
}
!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"func-sig"}

;--- negative.ll
define void @f() "cfi_type_hash"="-1" {
  ret void
}
!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"func-sig"}

;--- out-of-range.ll
define void @f() "cfi_type_hash"="1048576" {
  ret void
}
!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"func-sig"}