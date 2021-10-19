; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

declare i32 @bar(i32, i32)

define void @movep(i32 %a, i32 %b, i32 %c, i32 %d) {
; CHECK: movep $s1, $s0, $a0, $a1
; CHECK: movep $a0, $a1, $a2, $a3
  call i32 @bar(i32 %c, i32 %d)
; CHECK: movep $a0, $a1, $s1, $s0
  call i32 @bar(i32 %a, i32 %b)
  ret void
}

define i32 @movepnot(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i32 %g, i32 %h) {
; CHECK-NOT: movep
; CHECK: move
; CHECK: move
  %call = tail call i32 @bar(i32 %g, i32 %h)
  ret i32 %call
}
