; RUN: opt -S -instcombine < %s | FileCheck %s

; Checking for a crash

declare void @use.i1(i1 %val)
declare void @use.i64(i64 %val)

define i64 @f(i32 %x) {
; CHECK-LABEL: @f(
 entry:
  %x.wide = sext i32 %x to i64
  %minus.x = sub i32 0, %x
  %minus.x.wide = sext i32 %minus.x to i64
  %c = icmp slt i32 %x, 0
  %val = select i1 %c, i64 %x.wide, i64 %minus.x.wide
  call void @use.i1(i1 %c)
  call void @use.i64(i64 %x.wide)
  ret i64 %val
; CHECK: ret i64 %val
}
