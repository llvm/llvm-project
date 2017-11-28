; RUN: opt -instcombine -S < %s | FileCheck %s

; <rdar://problem/8606771>
define i32 @main(i32 %argc) {
; CHECK-LABEL: @main(
; CHECK-NEXT:    [[TMP3151:%.*]] = trunc i32 %argc to i8
; CHECK-NEXT:    [[TMP1:%.*]] = shl i8 [[TMP3151]], 5
; CHECK-NEXT:    [[TMP4126:%.*]] = and i8 [[TMP1]], 64
; CHECK-NEXT:    [[TMP4127:%.*]] = xor i8 [[TMP4126]], 64
; CHECK-NEXT:    [[TMP4086:%.*]] = zext i8 [[TMP4127]] to i32
; CHECK-NEXT:    ret i32 [[TMP4086]]
;
  %tmp3151 = trunc i32 %argc to i8
  %tmp3161 = or i8 %tmp3151, -17
  %tmp3162 = and i8 %tmp3151, 122
  %tmp3163 = xor i8 %tmp3162, -17
  %tmp4114 = shl i8 %tmp3163, 6
  %tmp4115 = xor i8 %tmp4114, %tmp3163
  %tmp4120 = xor i8 %tmp3161, %tmp4115
  %tmp4126 = lshr i8 %tmp4120, 7
  %tmp4127 = mul i8 %tmp4126, 64
  %tmp4086 = zext i8 %tmp4127 to i32
  ret i32 %tmp4086
}

; rdar://8739316
define i8 @foo(i8 %arg, i8 %arg1) {
; CHECK-LABEL: @foo(
; CHECK-NEXT:    [[TMP:%.*]] = shl i8 %arg, 7
; CHECK-NEXT:    [[TMP2:%.*]] = and i8 %arg1, 84
; CHECK-NEXT:    [[TMP3:%.*]] = and i8 %arg1, -118
; CHECK-NEXT:    [[TMP4:%.*]] = and i8 %arg1, 33
; CHECK-NEXT:    [[TMP5:%.*]] = sub nsw i8 40, [[TMP2]]
; CHECK-NEXT:    [[TMP6:%.*]] = and i8 [[TMP5]], 84
; CHECK-NEXT:    [[TMP7:%.*]] = or i8 [[TMP4]], [[TMP6]]
; CHECK-NEXT:    [[TMP8:%.*]] = xor i8 [[TMP]], [[TMP3]]
; CHECK-NEXT:    [[TMP9:%.*]] = or i8 [[TMP7]], [[TMP8]]
; CHECK-NEXT:    [[TMP10:%.*]] = lshr i8 [[TMP8]], 7
; CHECK-NEXT:    [[TMP11:%.*]] = shl nuw nsw i8 [[TMP10]], 5
; CHECK-NEXT:    [[TMP12:%.*]] = xor i8 [[TMP11]], [[TMP9]]
; CHECK-NEXT:    ret i8 [[TMP12]]
;
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
  %tmp12 = xor i8 %tmp11, %tmp9
  ret i8 %tmp12
}

