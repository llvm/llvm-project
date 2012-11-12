; RUN: llc < %s -march=ppc32 -mtriple=powerpc-unknown-linux-gnu | FileCheck %s

declare void @bar(i64 %x, i64 %y)

; CHECK: li 3, 0
; CHECK: li 4, 2
; CHECK: li 5, 0
; CHECK: li 6, 3

define void @foo() {
  call void @bar(i64 2, i64 3)
  ret void
}
