; RUN: sed 's/func-sig/bogus/' %s | not --crash llc --mtriple=riscv32 -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: sed 's/func-sig/bogus/' %s | not --crash llc --mtriple=riscv64 -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID

; (func-sig is replaced with bogus so it becomes invalid)
; INVALID: invalid RISC-V Zicfilp label scheme

!llvm.module.flags = !{!0, !1}

!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"func-sig"}