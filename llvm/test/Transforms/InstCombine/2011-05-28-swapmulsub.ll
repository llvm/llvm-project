; RUN: opt -S -instcombine < %s | FileCheck %s

target triple = "x86_64-apple-macosx10.6.6"

define zeroext i16 @foo1(i32 %on_off) {
; CHECK-LABEL: @foo1(
; CHECK-NEXT:    [[ON_OFF_TR:%.*]] = trunc i32 %on_off to i16
; CHECK-NEXT:    [[TMP1:%.*]] = shl i16 [[ON_OFF_TR]], 1
; CHECK-NEXT:    [[CONV:%.*]] = add i16 [[TMP1]], -2
; CHECK-NEXT:    ret i16 [[CONV]]
;
  %on_off.addr = alloca i32, align 4
  %a = alloca i32, align 4
  store i32 %on_off, i32* %on_off.addr, align 4
  %tmp = load i32, i32* %on_off.addr, align 4
  %sub = sub i32 1, %tmp
  %mul = mul i32 %sub, -2
  store i32 %mul, i32* %a, align 4
  %tmp1 = load i32, i32* %a, align 4
  %conv = trunc i32 %tmp1 to i16
  ret i16 %conv
}

define zeroext i16 @foo2(i32 %on_off, i32 %q) {
; CHECK-LABEL: @foo2(
; CHECK-NEXT:    [[SUBA:%.*]] = sub i32 %on_off, %q
; CHECK-NEXT:    [[SUBA_TR:%.*]] = trunc i32 [[SUBA]] to i16
; CHECK-NEXT:    [[CONV:%.*]] = shl i16 [[SUBA_TR]], 2
; CHECK-NEXT:    ret i16 [[CONV]]
;
  %on_off.addr = alloca i32, align 4
  %q.addr = alloca i32, align 4
  %a = alloca i32, align 4
  store i32 %on_off, i32* %on_off.addr, align 4
  store i32 %q, i32* %q.addr, align 4
  %tmp = load i32, i32* %q.addr, align 4
  %tmp1 = load i32, i32* %on_off.addr, align 4
  %sub = sub i32 %tmp, %tmp1
  %mul = mul i32 %sub, -4
  store i32 %mul, i32* %a, align 4
  %tmp2 = load i32, i32* %a, align 4
  %conv = trunc i32 %tmp2 to i16
  ret i16 %conv
}

define zeroext i16 @foo3(i32 %on_off) {
; CHECK-LABEL: @foo3(
; CHECK-NEXT:    [[ON_OFF_TR:%.*]] = trunc i32 %on_off to i16
; CHECK-NEXT:    [[TMP1:%.*]] = shl i16 [[ON_OFF_TR]], 2
; CHECK-NEXT:    [[CONV:%.*]] = add i16 [[TMP1]], -28
; CHECK-NEXT:    ret i16 [[CONV]]
;
  %on_off.addr = alloca i32, align 4
  %a = alloca i32, align 4
  store i32 %on_off, i32* %on_off.addr, align 4
  %tmp = load i32, i32* %on_off.addr, align 4
  %sub = sub i32 7, %tmp
  %mul = mul i32 %sub, -4
  store i32 %mul, i32* %a, align 4
  %tmp1 = load i32, i32* %a, align 4
  %conv = trunc i32 %tmp1 to i16
  ret i16 %conv
}

