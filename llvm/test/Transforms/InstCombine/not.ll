; RUN: opt < %s -instcombine -S | FileCheck %s

define i32 @test1(i32 %A) {
; CHECK-LABEL: @test1(
; CHECK-NEXT:    ret i32 %A
;
  %B = xor i32 %A, -1
  %C = xor i32 %B, -1
  ret i32 %C
}

define i1 @invert_icmp(i32 %A, i32 %B) {
; CHECK-LABEL: @invert_icmp(
; CHECK-NEXT:    [[CMP:%.*]] = icmp sgt i32 %A, %B
; CHECK-NEXT:    ret i1 [[CMP]]
;
  %cmp = icmp sle i32 %A, %B
  %not = xor i1 %cmp, true
  ret i1 %not
}

; PR1570

define i1 @invert_fcmp(float %X, float %Y) {
; CHECK-LABEL: @invert_fcmp(
; CHECK-NEXT:    [[CMP:%.*]] = fcmp uge float %X, %Y
; CHECK-NEXT:    ret i1 [[CMP]]
;
  %cmp = fcmp olt float %X, %Y
  %not = xor i1 %cmp, true
  ret i1 %not
}

; PR2298

define zeroext i8 @test6(i32 %a, i32 %b) {
; CHECK-LABEL: @test6(
; CHECK-NEXT:    [[TMP3:%.*]] = icmp slt i32 %b, %a
; CHECK-NEXT:    [[RETVAL67:%.*]] = zext i1 [[TMP3]] to i8
; CHECK-NEXT:    ret i8 [[RETVAL67]]
;
  %tmp1not = xor i32 %a, -1
  %tmp2not = xor i32 %b, -1
  %tmp3 = icmp slt i32 %tmp1not, %tmp2not
  %retval67 = zext i1 %tmp3 to i8
  ret i8 %retval67
}

define <2 x i1> @test7(<2 x i32> %A, <2 x i32> %B) {
; CHECK-LABEL: @test7(
; CHECK-NEXT:    [[COND:%.*]] = icmp sgt <2 x i32> %A, %B
; CHECK-NEXT:    ret <2 x i1> [[COND]]
;
  %cond = icmp sle <2 x i32> %A, %B
  %Ret = xor <2 x i1> %cond, <i1 true, i1 true>
  ret <2 x i1> %Ret
}

define i32 @not_ashr_not(i32 %A, i32 %B) {
; CHECK-LABEL: @not_ashr_not(
; CHECK-NEXT:    [[NOT2:%.*]] = ashr i32 %A, %B
; CHECK-NEXT:    ret i32 [[NOT2]]
;
  %not1 = xor i32 %A, -1
  %ashr = ashr i32 %not1, %B
  %not2 = xor i32 %ashr, -1
  ret i32 %not2
}

define i8 @not_ashr_const(i8 %x) {
; CHECK-LABEL: @not_ashr_const(
; CHECK-NEXT:    [[NOT:%.*]] = lshr i8 41, %x
; CHECK-NEXT:    ret i8 [[NOT]]
;
  %shr = ashr i8 -42, %x
  %not = xor i8 %shr, -1
  ret i8 %not
}

define <2 x i8> @not_ashr_const_splat(<2 x i8> %x) {
; CHECK-LABEL: @not_ashr_const_splat(
; CHECK-NEXT:    [[NOT:%.*]] = lshr <2 x i8> <i8 41, i8 41>, %x
; CHECK-NEXT:    ret <2 x i8> [[NOT]]
;
  %shr = ashr <2 x i8> <i8 -42, i8 -42>, %x
  %not = xor <2 x i8> %shr, <i8 -1, i8 -1>
  ret <2 x i8> %not
}

; We can't get rid of the 'not' on a logical shift of a negative constant.

define i8 @not_lshr_const_negative(i8 %x) {
; CHECK-LABEL: @not_lshr_const_negative(
; CHECK-NEXT:    [[SHR:%.*]] = lshr i8 -42, %x
; CHECK-NEXT:    [[NOT:%.*]] = xor i8 [[SHR]], -1
; CHECK-NEXT:    ret i8 [[NOT]]
;
  %shr = lshr i8 -42, %x
  %not = xor i8 %shr, -1
  ret i8 %not
}

define i8 @not_lshr_const(i8 %x) {
; CHECK-LABEL: @not_lshr_const(
; CHECK-NEXT:    [[NOT:%.*]] = ashr i8 -43, %x
; CHECK-NEXT:    ret i8 [[NOT]]
;
  %shr = lshr i8 42, %x
  %not = xor i8 %shr, -1
  ret i8 %not
}

define <2 x i8> @not_lshr_const_splat(<2 x i8> %x) {
; CHECK-LABEL: @not_lshr_const_splat(
; CHECK-NEXT:    [[NOT:%.*]] = ashr <2 x i8> <i8 -43, i8 -43>, %x
; CHECK-NEXT:    ret <2 x i8> [[NOT]]
;
  %shr = lshr <2 x i8> <i8 42, i8 42>, %x
  %not = xor <2 x i8> %shr, <i8 -1, i8 -1>
  ret <2 x i8> %not
}

