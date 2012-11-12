; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

define i32 @f1(i32 %a, i32 %b, i32 %c) {
; CHECK: f1:
; CHECK: muls r0, r1, r0
    %tmp = mul i32 %a, %b
    ret i32 %tmp
}

%struct.CMPoint = type { %struct.Point, float, float, [5 x float] }
%struct.Point = type { float, float }

define %struct.CMPoint* @t1(i32 %i, i32 %j, i32 %n, %struct.CMPoint* %thePoints) nounwind readnone ssp {
entry:
; CHECK: t1:
; CHECK: mla     r0, r2, r0, r1
; CHECK: add.w   r0, r0, r0, lsl #3
; CHECL: add.w   r0, r3, r0, lsl #2
  %mul = mul i32 %n, %i
  %add = add i32 %mul, %j
  %0 = ptrtoint %struct.CMPoint* %thePoints to i32
  %mul5 = mul i32 %add, 36
  %add6 = add i32 %mul5, %0
  %1 = inttoptr i32 %add6 to %struct.CMPoint*
  ret %struct.CMPoint* %1
}
