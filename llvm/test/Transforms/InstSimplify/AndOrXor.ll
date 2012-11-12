; RUN: opt < %s -instsimplify -S | FileCheck %s

define i64 @pow2(i32 %x) {
; CHECK: @pow2
  %negx = sub i32 0, %x
  %x2 = and i32 %x, %negx
  %e = zext i32 %x2 to i64
  %nege = sub i64 0, %e
  %e2 = and i64 %e, %nege
  ret i64 %e2
; CHECK: ret i64 %e
}

define i64 @pow2b(i32 %x) {
; CHECK: @pow2b
  %sh = shl i32 2, %x
  %e = zext i32 %sh to i64
  %nege = sub i64 0, %e
  %e2 = and i64 %e, %nege
  ret i64 %e2
; CHECK: ret i64 %e
}
