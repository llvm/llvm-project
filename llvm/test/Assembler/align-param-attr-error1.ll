; RUN: not llvm-as < %s 2>&1 | FileCheck %s
; Test parse errors when using form of align attribute with parentheses

; CHECK:  <stdin>:[[@LINE+1]]:42: error: expected '{' in function body
define void @missing_Lparen(ptr align 4) %ptr) {
  ret void
}
