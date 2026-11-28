; RUN: split-file %s %t
; RUN: not --crash llc -mtriple=riscv32 -verify-machineinstrs < %t/case1.ll 2>&1 | FileCheck %s --check-prefix=ERROR-FUNCSIG
; RUN: not --crash llc -mtriple=riscv32 -verify-machineinstrs < %t/case2.ll 2>&1 | FileCheck %s --check-prefix=ERROR-FUNCSIG
; RUN: not --crash llc -mtriple=riscv32 -verify-machineinstrs < %t/case3.ll 2>&1 | FileCheck %s --check-prefix=ERROR-UNLABELED
; RUN: not --crash llc -mtriple=riscv32 -verify-machineinstrs < %t/case4.ll 2>&1 | FileCheck %s --check-prefix=ERROR-UNLABELED

; ERROR-FUNCSIG: supplied RISCV landing pad label is 0 but selected scheme is 'func-sig'
; ERROR-UNLABELED: supplied RISCV landing pad label is not 0 but selected scheme is 'unlabeled'
;; note: one of the possible misconfigs does not currently error. see: zicfilp-func-sig-missing-label.ll

;--- case1.ll
; case 1: cfi_type_hash is 0
define i32 @test_call_indirect(ptr %a, i32 %b) nounwind "cfi_type_hash"="0" {
  %1 = call i32 %a(i32 %b), !riscv_lpad_label !2
  ret i32 %1
}
!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"func-sig"}
!2 = !{i32 1234}

;--- case2.ll
; case 2: riscv_lpad_label is 0
define i32 @test_call_indirect2(ptr %a, i32 %b) nounwind "cfi_type_hash"="1234" {
  %1 = call i32 %a(i32 %b), !riscv_lpad_label !3
  ret i32 %1
}
!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"func-sig"}
!3 = !{i32 0}

;--- case3.ll
; case 3: cfi_type_hash is non-0
define i32 @test_call_indirect(ptr %a, i32 %b) nounwind "cfi_type_hash"="778899" {
  %1 = call i32 %a(i32 %b)
  ret i32 %1
}
!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"unlabeled"}
!2 = !{i32 112233}

;--- case4.ll
; case 4: riscv_lpad_label is non-0
define i32 @test_call_indirect2(ptr %a, i32 %b) nounwind "cfi_type_hash"="0" {
  %1 = call i32 %a(i32 %b), !riscv_lpad_label !2
  ret i32 %1
}
!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"unlabeled"}
!2 = !{i32 112233}
