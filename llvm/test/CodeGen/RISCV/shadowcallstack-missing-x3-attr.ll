; RUN: not --crash llc -mtriple=riscv32 < %s  2>&1 \
; RUN:   | FileCheck --check-prefix=X3ERR %s
; RUN: not --crash llc -mtriple=riscv64 < %s  2>&1 \
; RUN:   | FileCheck --check-prefix=X3ERR %s

;; Its safe for Zicfiss not to set x3-scs.
; RUN: llc -mtriple=riscv64 < %s -mattr=+experimental-zicfiss \
; RUN:   | FileCheck --check-prefix=NOX3ERR %s

;; It isn't safe w/ forced-sw-shadow-stack, though
; RUN: not --crash llc -mtriple=riscv64 <%s -mattr=+experimental-zicfiss,forced-sw-shadow-stack 2>&1 \
; RUN:   | FileCheck --check-prefix=X3ERR %s

; X3ERR: LLVM ERROR: Cannot use the software based RISCV shadow call stack without setting the ABI tag `+x3-scs`.
; NOX3ERR-NOT: LLVM ERROR

declare i32 @bar()

define i32 @f1() shadowcallstack {
  %res = call i32 @bar()
  %res1 = add i32 %res, 1
  ret i32 %res
}
