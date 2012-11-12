; RUN: llc < %s -mtriple=thumb-apple-darwin | FileCheck %s
; rdar://7268481

define void @t(i8* %a, ...) nounwind {
; CHECK:      t:
; CHECK:      pop {r3}
; CHECK-NEXT: add sp, #12
; CHECK-NEXT: bx r3
entry:
  %a.addr = alloca i8*
  store i8* %a, i8** %a.addr
  ret void
}
