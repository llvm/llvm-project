; RUN: llc < %s -march=x86 | FileCheck %s

define i32 @t1(i8 zeroext %x) nounwind readnone ssp {
entry:
; CHECK: t1:
; CHECK: shll
; CHECK-NOT: movzwl
; CHECK: ret
  %0 = zext i8 %x to i16
  %1 = shl i16 %0, 5
  %2 = zext i16 %1 to i32
  ret i32 %2
}

define i32 @t2(i8 zeroext %x) nounwind readnone ssp {
entry:
; CHECK: t2:
; CHECK: shrl
; CHECK-NOT: movzwl
; CHECK: ret
  %0 = zext i8 %x to i16
  %1 = lshr i16 %0, 3
  %2 = zext i16 %1 to i32
  ret i32 %2
}
