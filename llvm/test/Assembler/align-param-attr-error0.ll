; RUN: not llvm-as < %s 2>&1 | FileCheck %s
; Test parse errors when using form of align attribute with parentheses

; CHECK:  <stdin>:[[@LINE+1]]:38: error: expected ')'
define void @missing_rparen(ptr align(4 %ptr) {
  ret void
}
