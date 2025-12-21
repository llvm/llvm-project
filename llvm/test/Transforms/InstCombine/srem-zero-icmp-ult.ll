; RUN: opt -passes=instcombine -S < %s | FileCheck %s

define i64 @f(i1 %c) {
entry:
  %0 = select i1 %c, i64 3, i64 0
  %1 = trunc i64 %0 to i8
  %2 = sdiv i8 -1, %1
  %3 = srem i8 -1, %2
  %4 = icmp ult i8 %1, %3
  %5 = zext i1 %4 to i64
  %6 = select i1 false, i64 %0, i64 %5
  ret i64 %6
}

; CHECK-LABEL: @f(
; CHECK: ret i64 poison
