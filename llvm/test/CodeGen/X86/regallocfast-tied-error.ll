; RUN: not llc -mtriple=x86_64-linux -O0 -regalloc-fast-tied=false < %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=x86_64-linux -O0 -regalloc-fast-tied=true < %s 2>&1 | FileCheck %s

;; The last output cannot be allocated and gets an error assignment to the
;; register of the tied output. Freeing the defs must not release the tied
;; output's register before its tied use is allocated.

; CHECK: error: inline assembly requires more registers than available

define void @tied_def_shares_error_assignment(ptr %p) nounwind {
  %in = load i64, ptr %p
  %r = call { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } asm "", "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,0"(i64 %in)
  %out = extractvalue { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } %r, 0
  store i64 %out, ptr %p
  ret void
}
