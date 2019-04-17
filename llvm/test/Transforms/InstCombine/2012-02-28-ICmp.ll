; RUN: opt < %s -instcombine -S | FileCheck %s
; <rdar://problem/10803154>

; There should be no transformation.
; CHECK: %a = trunc i32 %x to i8
; CHECK: %b = icmp ne i8 %a, 0
; CHECK: %c = and i32 %x, 16711680
; CHECK: %d = icmp ne i32 %c, 0
; CHECK: %e = and i1 %b, %d
; CHECK: ret i1 %e

define i1 @f1(i32 %x) {
  %a = trunc i32 %x to i8
  %b = icmp ne i8 %a, 0
  %c = and i32 %x, 16711680
  %d = icmp ne i32 %c, 0
  %e = and i1 %b, %d
  ret i1 %e
}
