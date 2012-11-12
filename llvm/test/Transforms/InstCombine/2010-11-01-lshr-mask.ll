; RUN: opt -instcombine -S < %s | FileCheck %s

; <rdar://problem/8606771>
; CHECK: @main
define i32 @main(i32 %argc) nounwind ssp {
entry:
  %tmp3151 = trunc i32 %argc to i8
; CHECK: %tmp3163 = shl i8 %tmp3162, 6
; CHECK: and i8 %tmp3163, 64
; CHECK-NOT: shl
; CHECK-NOT: shr
  %tmp3161 = or i8 %tmp3151, -17
  %tmp3162 = and i8 %tmp3151, 122
  %tmp3163 = xor i8 %tmp3162, -17
  %tmp4114 = shl i8 %tmp3163, 6
  %tmp4115 = xor i8 %tmp4114, %tmp3163
  %tmp4120 = xor i8 %tmp3161, %tmp4115
  %tmp4126 = lshr i8 %tmp4120, 7
  %tmp4127 = mul i8 %tmp4126, 64
  %tmp4086 = zext i8 %tmp4127 to i32
; CHECK: ret i32
  ret i32 %tmp4086
}

; rdar://8739316
; CHECK: @foo
define i8 @foo(i8 %arg, i8 %arg1) nounwind {
bb:
  %tmp = shl i8 %arg, 7
  %tmp2 = and i8 %arg1, 84
  %tmp3 = and i8 %arg1, -118
  %tmp4 = and i8 %arg1, 33
  %tmp5 = sub i8 -88, %tmp2
  %tmp6 = and i8 %tmp5, 84
  %tmp7 = or i8 %tmp4, %tmp6
  %tmp8 = xor i8 %tmp, %tmp3
  %tmp9 = or i8 %tmp7, %tmp8
  %tmp10 = lshr i8 %tmp8, 7
  %tmp11 = shl i8 %tmp10, 5

; CHECK: %tmp10 = lshr i8 %tmp8, 7
; CHECK: %tmp11 = shl nuw nsw i8 %tmp10, 5

  %tmp12 = xor i8 %tmp11, %tmp9
  ret i8 %tmp12
}
