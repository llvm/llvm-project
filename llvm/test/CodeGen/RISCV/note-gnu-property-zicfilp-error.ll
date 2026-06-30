; RUN: sed 's/SCHEME/func-sig/' %s | not llc --mtriple=riscv32 -o /dev/null 2>&1 | FileCheck %s --check-prefix=FUNC-SIG
; RUN: sed 's/SCHEME/func-sig/' %s | not llc --mtriple=riscv64 -o /dev/null 2>&1 | FileCheck %s --check-prefix=FUNC-SIG

; RUN: sed 's/SCHEME/bogus/' %s | not --crash llc --mtriple=riscv32 -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: sed 's/SCHEME/bogus/' %s | not --crash llc --mtriple=riscv64 -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID

; FUNC-SIG: LLVM ERROR: the complete func-sig label scheme feature is not implemented yet

; INVALID: LLVM ERROR: invalid RISC-V Zicfilp label scheme

!llvm.module.flags = !{!0, !1}

!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"SCHEME"}
