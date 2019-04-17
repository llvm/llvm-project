; RUN: opt -S -instcombine < %s | FileCheck %s

define <4 x i32> @test_FoldShiftByConstant_CreateSHL(<4 x i32> %in) {
; CHECK-LABEL: @test_FoldShiftByConstant_CreateSHL(
; CHECK-NEXT:    [[VSHL_N:%.*]] = mul <4 x i32> %in, <i32 0, i32 -32, i32 0, i32 -32>
; CHECK-NEXT:    ret <4 x i32> [[VSHL_N]]
;
  %mul.i = mul <4 x i32> %in, <i32 0, i32 -1, i32 0, i32 -1>
  %vshl_n = shl <4 x i32> %mul.i, <i32 5, i32 5, i32 5, i32 5>
  ret <4 x i32> %vshl_n
}

define <8 x i16> @test_FoldShiftByConstant_CreateSHL2(<8 x i16> %in) {
; CHECK-LABEL: @test_FoldShiftByConstant_CreateSHL2(
; CHECK-NEXT:    [[VSHL_N:%.*]] = mul <8 x i16> %in, <i16 0, i16 -32, i16 0, i16 -32, i16 0, i16 -32, i16 0, i16 -32>
; CHECK-NEXT:    ret <8 x i16> [[VSHL_N]]
;
  %mul.i = mul <8 x i16> %in, <i16 0, i16 -1, i16 0, i16 -1, i16 0, i16 -1, i16 0, i16 -1>
  %vshl_n = shl <8 x i16> %mul.i, <i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5>
  ret <8 x i16> %vshl_n
}

define <16 x i8> @test_FoldShiftByConstant_CreateAnd(<16 x i8> %in0) {
; CHECK-LABEL: @test_FoldShiftByConstant_CreateAnd(
; CHECK-NEXT:    [[VSRA_N2:%.*]] = mul <16 x i8> %in0, <i8 33, i8 33, i8 33, i8 33, i8 33, i8 33, i8 33, i8 33, i8 33, i8 33, i8 33, i8 33, i8 33, i8 33, i8 33, i8 33>
; CHECK-NEXT:    [[VSHL_N:%.*]] = and <16 x i8> [[VSRA_N2]], <i8 -32, i8 -32, i8 -32, i8 -32, i8 -32, i8 -32, i8 -32, i8 -32, i8 -32, i8 -32, i8 -32, i8 -32, i8 -32, i8 -32, i8 -32, i8 -32>
; CHECK-NEXT:    ret <16 x i8> [[VSHL_N]]
;
  %vsra_n = ashr <16 x i8> %in0, <i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5>
  %tmp = add <16 x i8> %in0, %vsra_n
  %vshl_n = shl <16 x i8> %tmp, <i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5>
  ret <16 x i8> %vshl_n
}

define i32 @bar(i32 %x, i32 %y) {
; CHECK-LABEL: @bar(
; CHECK-NEXT:    [[B1:%.*]] = shl i32 %y, 4
; CHECK-NEXT:    [[A2:%.*]] = add i32 [[B1]], %x
; CHECK-NEXT:    [[C:%.*]] = and i32 [[A2]], -16
; CHECK-NEXT:    ret i32 [[C]]
;
  %a = lshr i32 %x, 4
  %b = add i32 %a, %y
  %c = shl i32 %b, 4
  ret i32 %c
}

define <2 x i32> @bar_v2i32(<2 x i32> %x, <2 x i32> %y) {
; CHECK-LABEL: @bar_v2i32(
; CHECK-NEXT:    [[B1:%.*]] = shl <2 x i32> %y, <i32 5, i32 5>
; CHECK-NEXT:    [[A2:%.*]] = add <2 x i32> [[B1]], %x
; CHECK-NEXT:    [[C:%.*]] = and <2 x i32> [[A2]], <i32 -32, i32 -32>
; CHECK-NEXT:    ret <2 x i32> [[C]]
;
  %a = lshr <2 x i32> %x, <i32 5, i32 5>
  %b = add <2 x i32> %a, %y
  %c = shl <2 x i32> %b, <i32 5, i32 5>
  ret <2 x i32> %c
}

define i32 @foo(i32 %x, i32 %y) {
; CHECK-LABEL: @foo(
; CHECK-NEXT:    [[C1:%.*]] = shl i32 %y, 4
; CHECK-NEXT:    [[X_MASK:%.*]] = and i32 %x, 128
; CHECK-NEXT:    [[D:%.*]] = add i32 [[X_MASK]], [[C1]]
; CHECK-NEXT:    ret i32 [[D]]
;
  %a = lshr i32 %x, 4
  %b = and i32 %a, 8
  %c = add i32 %b, %y
  %d = shl i32 %c, 4
  ret i32 %d
}

define <2 x i32> @foo_v2i32(<2 x i32> %x, <2 x i32> %y) {
; CHECK-LABEL: @foo_v2i32(
; CHECK-NEXT:    [[A:%.*]] = lshr <2 x i32> %x, <i32 4, i32 4>
; CHECK-NEXT:    [[B:%.*]] = and <2 x i32> [[A]], <i32 8, i32 8>
; CHECK-NEXT:    [[C:%.*]] = add <2 x i32> [[B]], %y
; CHECK-NEXT:    [[D:%.*]] = shl <2 x i32> [[C]], <i32 4, i32 4>
; CHECK-NEXT:    ret <2 x i32> [[D]]
;
  %a = lshr <2 x i32> %x, <i32 4, i32 4>
  %b = and <2 x i32> %a, <i32 8, i32 8>
  %c = add <2 x i32> %b, %y
  %d = shl <2 x i32> %c, <i32 4, i32 4>
  ret <2 x i32> %d
}

