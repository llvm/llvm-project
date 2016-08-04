; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=g4 | FileCheck %s

define void @test(i32* nocapture %x, i64* %xx, i32* %yp) nounwind uwtable ssp {
entry:
  %yy = load i32, i32* %yp
  %y = add i32 %yy, 1
  %z = zext i32 %y to i64
  %z2 = shl i64 %z, 32 
  store i64 %z2, i64* %xx, align 4
  ret void

; CHECK-LABEL: test:
; CHECK: sldi {{.*}}, {{.*}}, 32
; Note: it's okay if someday CodeGen gets smart enough to optimize out
; the shift.
}
