; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: int32_float_pair
; CHECK: movss %xmm0, 4(%rsi)
; CHECK: movl %edi, (%rsi)
define void @int32_float_pair(i32 %tmp1, float %tmp2, i64* %ref.tmp) {
entry:
  %t0 = bitcast float %tmp2 to i32
  %t1 = zext i32 %t0 to i64
  %t2 = shl nuw i64 %t1, 32
  %t3 = zext i32 %tmp1 to i64
  %t4 = or i64 %t2, %t3
  store i64 %t4, i64* %ref.tmp, align 8
  ret void
}

; CHECK-LABEL: float_int32_pair
; CHECK: movl %edi, 4(%rsi)
; CHECK: movss %xmm0, (%rsi)
define void @float_int32_pair(float %tmp1, i32 %tmp2, i64* %ref.tmp) {
entry:
  %t0 = bitcast float %tmp1 to i32
  %t1 = zext i32 %tmp2 to i64
  %t2 = shl nuw i64 %t1, 32
  %t3 = zext i32 %t0 to i64
  %t4 = or i64 %t2, %t3
  store i64 %t4, i64* %ref.tmp, align 8
  ret void
}

; CHECK-LABEL: int16_float_pair
; CHECK: movss %xmm0, 4(%rsi)
; CHECK: movzwl	%di, %eax
; CHECK: movl %eax, (%rsi)
define void @int16_float_pair(i16 signext %tmp1, float %tmp2, i64* %ref.tmp) {
entry:
  %t0 = bitcast float %tmp2 to i32
  %t1 = zext i32 %t0 to i64
  %t2 = shl nuw i64 %t1, 32
  %t3 = zext i16 %tmp1 to i64
  %t4 = or i64 %t2, %t3
  store i64 %t4, i64* %ref.tmp, align 8
  ret void
}

; CHECK-LABEL: int8_float_pair
; CHECK: movss %xmm0, 4(%rsi)
; CHECK: movzbl	%dil, %eax
; CHECK: movl %eax, (%rsi)
define void @int8_float_pair(i8 signext %tmp1, float %tmp2, i64* %ref.tmp) {
entry:
  %t0 = bitcast float %tmp2 to i32
  %t1 = zext i32 %t0 to i64
  %t2 = shl nuw i64 %t1, 32
  %t3 = zext i8 %tmp1 to i64
  %t4 = or i64 %t2, %t3
  store i64 %t4, i64* %ref.tmp, align 8
  ret void
}
