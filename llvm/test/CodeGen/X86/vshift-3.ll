; RUN: llc < %s -march=x86 -mattr=+sse2 | FileCheck %s

; test vector shifts converted to proper SSE2 vector shifts when the shift
; amounts are the same.

; Note that x86 does have ashr 

; shift1a can't use a packed shift
define void @shift1a(<2 x i64> %val, <2 x i64>* %dst) nounwind {
entry:
; CHECK: shift1a:
; CHECK: sarl
  %ashr = ashr <2 x i64> %val, < i64 32, i64 32 >
  store <2 x i64> %ashr, <2 x i64>* %dst
  ret void
}

define void @shift2a(<4 x i32> %val, <4 x i32>* %dst) nounwind {
entry:
; CHECK: shift2a:
; CHECK: psrad	$5
  %ashr = ashr <4 x i32> %val, < i32 5, i32 5, i32 5, i32 5 >
  store <4 x i32> %ashr, <4 x i32>* %dst
  ret void
}

define void @shift2b(<4 x i32> %val, <4 x i32>* %dst, i32 %amt) nounwind {
entry:
; CHECK: shift2b:
; CHECK: movd
; CHECK: psrad
  %0 = insertelement <4 x i32> undef, i32 %amt, i32 0
  %1 = insertelement <4 x i32> %0, i32 %amt, i32 1
  %2 = insertelement <4 x i32> %1, i32 %amt, i32 2
  %3 = insertelement <4 x i32> %2, i32 %amt, i32 3
  %ashr = ashr <4 x i32> %val, %3
  store <4 x i32> %ashr, <4 x i32>* %dst
  ret void
}

define void @shift3a(<8 x i16> %val, <8 x i16>* %dst) nounwind {
entry:
; CHECK: shift3a:
; CHECK: psraw	$5
  %ashr = ashr <8 x i16> %val, < i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5 >
  store <8 x i16> %ashr, <8 x i16>* %dst
  ret void
}

define void @shift3b(<8 x i16> %val, <8 x i16>* %dst, i16 %amt) nounwind {
entry:
; CHECK: shift3b:
; CHECK: movzwl
; CHECK: movd
; CHECK: psraw
  %0 = insertelement <8 x i16> undef, i16 %amt, i32 0
  %1 = insertelement <8 x i16> %0, i16 %amt, i32 1
  %2 = insertelement <8 x i16> %0, i16 %amt, i32 2
  %3 = insertelement <8 x i16> %0, i16 %amt, i32 3
  %4 = insertelement <8 x i16> %0, i16 %amt, i32 4
  %5 = insertelement <8 x i16> %0, i16 %amt, i32 5
  %6 = insertelement <8 x i16> %0, i16 %amt, i32 6
  %7 = insertelement <8 x i16> %0, i16 %amt, i32 7
  %ashr = ashr <8 x i16> %val, %7
  store <8 x i16> %ashr, <8 x i16>* %dst
  ret void
}
