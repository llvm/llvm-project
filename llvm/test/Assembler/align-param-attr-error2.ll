; RUN: not llvm-as < %s 2>&1 | FileCheck %s
; Test parse errors when using form of align attribute with parentheses

; CHECK:  <stdin>:[[@LINE+1]]:39: error: expected integer
define void @missing_value(ptr align () %ptr) {
  ret void
}
