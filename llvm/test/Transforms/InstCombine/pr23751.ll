; RUN: opt -passes=instcombine -S < %s | FileCheck %s

@d = common global i32 0, align 4

define i1 @f(i8 zeroext %p) #1 {
; CHECK-NOT: ret i1 false
  %1 = zext i8 %p to i32
  %2 = load i32, i32* @d, align 4
  %3 = or i32 %2, -2
  %4 = add nsw i32 %3, %1
  %5 = icmp ugt i32 %1, %4
  ret i1 %5
}
