; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"


define void @one(i64 %a, i64 %b, ptr %p1, ptr %p2) {
; CHECK: cvt.s64.s8
; CHECK: cvt.s64.s8
entry:
  %sext = shl i64 %a, 56
  %conv1 = ashr exact i64 %sext, 56
  %sext1 = shl i64 %b, 56
  %conv4 = ashr exact i64 %sext1, 56
  %shr = ashr i64 %a, 16
  %shr9 = ashr i64 %b, 16
  %add = add nsw i64 %conv4, %conv1
  store i64 %add, ptr %p1, align 8
  %add17 = add nsw i64 %shr9, %shr
  store i64 %add17, ptr %p2, align 8
  ret void
}


define void @two(i64 %a, i64 %b, ptr %p1, ptr %p2) {
entry:
; CHECK: cvt.s64.s32
; CHECK: cvt.s64.s32
  %sext = shl i64 %a, 32
  %conv1 = ashr exact i64 %sext, 32
  %sext1 = shl i64 %b, 32
  %conv4 = ashr exact i64 %sext1, 32
  %shr = ashr i64 %a, 16
  %shr9 = ashr i64 %b, 16
  %add = add nsw i64 %conv4, %conv1
  store i64 %add, ptr %p1, align 8
  %add17 = add nsw i64 %shr9, %shr
  store i64 %add17, ptr %p2, align 8
  ret void
}


define void @three(i64 %a, i64 %b, ptr %p1, ptr %p2) {
entry:
; CHECK: cvt.s64.s16
; CHECK: cvt.s64.s16
  %sext = shl i64 %a, 48
  %conv1 = ashr exact i64 %sext, 48
  %sext1 = shl i64 %b, 48
  %conv4 = ashr exact i64 %sext1, 48
  %shr = ashr i64 %a, 16
  %shr9 = ashr i64 %b, 16
  %add = add nsw i64 %conv4, %conv1
  store i64 %add, ptr %p1, align 8
  %add17 = add nsw i64 %shr9, %shr
  store i64 %add17, ptr %p2, align 8
  ret void
}


define void @four(i32 %a, i32 %b, ptr %p1, ptr %p2) {
entry:
; CHECK: cvt.s32.s8
; CHECK: cvt.s32.s8
  %sext = shl i32 %a, 24
  %conv1 = ashr exact i32 %sext, 24
  %sext1 = shl i32 %b, 24
  %conv4 = ashr exact i32 %sext1, 24
  %shr = ashr i32 %a, 16
  %shr9 = ashr i32 %b, 16
  %add = add nsw i32 %conv4, %conv1
  store i32 %add, ptr %p1, align 4
  %add17 = add nsw i32 %shr9, %shr
  store i32 %add17, ptr %p2, align 4
  ret void
}


define void @five(i32 %a, i32 %b, ptr %p1, ptr %p2) {
entry:
; CHECK: cvt.s32.s16
; CHECK: cvt.s32.s16
  %sext = shl i32 %a, 16
  %conv1 = ashr exact i32 %sext, 16
  %sext1 = shl i32 %b, 16
  %conv4 = ashr exact i32 %sext1, 16
  %shr = ashr i32 %a, 16
  %shr9 = ashr i32 %b, 16
  %add = add nsw i32 %conv4, %conv1
  store i32 %add, ptr %p1, align 4
  %add17 = add nsw i32 %shr9, %shr
  store i32 %add17, ptr %p2, align 4
  ret void
}


define void @six(i16 %a, i16 %b, ptr %p1, ptr %p2) {
entry:
; CHECK: cvt.s16.s8
; CHECK: cvt.s16.s8
  %sext = shl i16 %a, 8
  %conv1 = ashr exact i16 %sext, 8
  %sext1 = shl i16 %b, 8
  %conv4 = ashr exact i16 %sext1, 8
  %shr = ashr i16 %a, 8
  %shr9 = ashr i16 %b, 8
  %add = add nsw i16 %conv4, %conv1
  store i16 %add, ptr %p1, align 4
  %add17 = add nsw i16 %shr9, %shr
  store i16 %add17, ptr %p2, align 4
  ret void
}
