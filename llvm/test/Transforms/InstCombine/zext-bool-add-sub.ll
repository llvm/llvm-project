; RUN: opt < %s -instcombine -S | FileCheck %s
; rdar://11748024

define i32 @a(i1 zeroext %x, i1 zeroext %y) {
entry:
; CHECK: @a
; CHECK: [[TMP1:%.*]] = zext i1 %y to i32
; CHECK: [[TMP2:%.*]] = select i1 %x, i32 2, i32 1
; CHECK-NEXT: sub i32 [[TMP2]], [[TMP1]]
  %conv = zext i1 %x to i32
  %conv3 = zext i1 %y to i32
  %conv3.neg = sub i32 0, %conv3
  %sub = add i32 %conv, 1
  %add = add i32 %sub, %conv3.neg
  ret i32 %add
}
