; RUN: llc -mtriple=riscv64 < %s | FileCheck %s --match-full-lines
; RUN: llc -mtriple=riscv32 < %s | FileCheck %s --match-full-lines

declare void @extern_func()

; CHECK-LABEL: const:
; CHECK-NEXT:    .word   extern_func@PLT-const

;; Note that for riscv32, the ptrtoint will actually upcast the ptr it to an
;; oversized 64-bit pointer that eventually gets truncated. This isn't needed
;; for riscv32, but this unifies the RV64 and RV32 test cases.
@const = dso_local constant i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @extern_func to i64), i64 ptrtoint (ptr @const to i64)) to i32)
