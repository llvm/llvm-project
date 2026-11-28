; RUN: split-file %s %t
; RUN: not --crash llc -mtriple=riscv32 -verify-machineinstrs < %t/empty-tuple.ll 2>&1 \
; RUN:   | FileCheck %s --check-prefix=WRONG-OPERAND-COUNT
; RUN: not --crash llc -mtriple=riscv32 -verify-machineinstrs < %t/two-operands.ll 2>&1 \
; RUN:   | FileCheck %s --check-prefix=WRONG-OPERAND-COUNT
; RUN: not --crash llc -mtriple=riscv32 -verify-machineinstrs < %t/string-operand.ll 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NOT-CONSTANT
; RUN: not --crash llc -mtriple=riscv32 -verify-machineinstrs < %t/i1-operand.ll 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NOT-I32
; RUN: not --crash llc -mtriple=riscv32 -verify-machineinstrs < %t/i8-negative-operand.ll 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NOT-I32
; RUN: not --crash llc -mtriple=riscv32 -verify-machineinstrs < %t/i64-operand.ll 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NOT-I32
; RUN: not --crash llc -mtriple=riscv32 -verify-machineinstrs < %t/i128-operand.ll 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NOT-I32
; RUN: not --crash llc -mtriple=riscv32 -verify-machineinstrs < %t/out-of-range.ll 2>&1 \
; RUN:   | FileCheck %s --check-prefix=OUT-OF-RANGE

; Exercises RISCVMachineFunctionInfo::getZicfilpCallSiteLabel's validation of
; malformed !riscv_lpad_label metadata on an indirect call. The
; missing-metadata case is tolerated for now (see comments in source)

; WRONG-OPERAND-COUNT: LLVM ERROR: riscv_lpad_label metadata must have exactly one operand
; NOT-CONSTANT: LLVM ERROR: riscv_lpad_label metadata operand must be a constant
; NOT-I32: LLVM ERROR: riscv_lpad_label metadata operand must be i32
; OUT-OF-RANGE: LLVM ERROR: riscv_lpad_label metadata value out of range

;--- empty-tuple.ll
define i32 @test_call_indirect(ptr %a, i32 %b) nounwind "cfi_type_hash"="778899" {
  %1 = call i32 %a(i32 %b), !riscv_lpad_label !2
  ret i32 %1
}
!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"func-sig"}
!2 = !{}

;--- two-operands.ll
define i32 @test_call_indirect(ptr %a, i32 %b) nounwind "cfi_type_hash"="778899" {
  %1 = call i32 %a(i32 %b), !riscv_lpad_label !2
  ret i32 %1
}
!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"func-sig"}
!2 = !{i32 1, i32 2}

;--- string-operand.ll
define i32 @test_call_indirect(ptr %a, i32 %b) nounwind "cfi_type_hash"="778899" {
  %1 = call i32 %a(i32 %b), !riscv_lpad_label !2
  ret i32 %1
}
!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"func-sig"}
!2 = !{!"x"}

;--- i1-operand.ll
define i32 @test_call_indirect(ptr %a, i32 %b) nounwind "cfi_type_hash"="778899" {
  %1 = call i32 %a(i32 %b), !riscv_lpad_label !2
  ret i32 %1
}
!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"func-sig"}
!2 = !{i1 1}

;--- i8-negative-operand.ll
define i32 @test_call_indirect(ptr %a, i32 %b) nounwind "cfi_type_hash"="778899" {
  %1 = call i32 %a(i32 %b), !riscv_lpad_label !2
  ret i32 %1
}
!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"func-sig"}
!2 = !{i8 -1}

;--- i64-operand.ll
define i32 @test_call_indirect(ptr %a, i32 %b) nounwind "cfi_type_hash"="778899" {
  %1 = call i32 %a(i32 %b), !riscv_lpad_label !2
  ret i32 %1
}
!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"func-sig"}
!2 = !{i64 1}

;--- i128-operand.ll
define i32 @test_call_indirect(ptr %a, i32 %b) nounwind "cfi_type_hash"="778899" {
  %1 = call i32 %a(i32 %b), !riscv_lpad_label !2
  ret i32 %1
}
!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"func-sig"}
!2 = !{i128 1}

;--- out-of-range.ll
define i32 @test_call_indirect(ptr %a, i32 %b) nounwind "cfi_type_hash"="778899" {
  %1 = call i32 %a(i32 %b), !riscv_lpad_label !2
  ret i32 %1
}
!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"func-sig"}
!2 = !{i32 1048576}