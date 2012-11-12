; RUN: llc < %s -march=x86-64 | FileCheck %s
; rdar://7570931

define i64 @foo(i64 %a, i64 %b) nounwind {
; CHECK: foo:
; CHECK: leal
; CHECK-NOT: movl
; CHECK: ret
  %c = add i64 %a, %b
  %d = trunc i64 %c to i32
  %e = zext i32 %d to i64
  ret i64 %e
}
