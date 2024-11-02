; Test insertions of byte-swapped memory values into a nonzero index of an undef.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z15 | FileCheck %s

declare i16 @llvm.bswap.i16(i16)
declare i32 @llvm.bswap.i32(i32)
declare i64 @llvm.bswap.i64(i64)
declare <8 x i16> @llvm.bswap.v8i16(<8 x i16>)
declare <4 x i32> @llvm.bswap.v4i32(<4 x i32>)
declare <2 x i64> @llvm.bswap.v2i64(<2 x i64>)

; Test v8i16 insertion into an undef, with an arbitrary index.
define <8 x i16> @f1(ptr %ptr) {
; CHECK-LABEL: f1:
; CHECK: vlbrreph %v24, 0(%r2)
; CHECK-NEXT: br %r14
  %val = load i16, ptr %ptr
  %swap = call i16 @llvm.bswap.i16(i16 %val)
  %ret = insertelement <8 x i16> undef, i16 %swap, i32 5
  ret <8 x i16> %ret
}

; Test v8i16 insertion into an undef, using a vector bswap.
define <8 x i16> @f2(ptr %ptr) {
; CHECK-LABEL: f2:
; CHECK: vlbrreph %v24, 0(%r2)
; CHECK-NEXT: br %r14
  %val = load i16, ptr %ptr
  %insert = insertelement <8 x i16> undef, i16 %val, i32 5
  %ret = call <8 x i16> @llvm.bswap.v8i16(<8 x i16> %insert)
  ret <8 x i16> %ret
}

; Test v4i32 insertion into an undef, with an arbitrary index.
define <4 x i32> @f3(ptr %ptr) {
; CHECK-LABEL: f3:
; CHECK: vlbrrepf %v24, 0(%r2)
; CHECK-NEXT: br %r14
  %val = load i32, ptr %ptr
  %swap = call i32 @llvm.bswap.i32(i32 %val)
  %ret = insertelement <4 x i32> undef, i32 %swap, i32 2
  ret <4 x i32> %ret
}

; Test v4i32 insertion into an undef, using a vector bswap.
define <4 x i32> @f4(ptr %ptr) {
; CHECK-LABEL: f4:
; CHECK: vlbrrepf %v24, 0(%r2)
; CHECK-NEXT: br %r14
  %val = load i32, ptr %ptr
  %insert = insertelement <4 x i32> undef, i32 %val, i32 2
  %ret = call <4 x i32> @llvm.bswap.v4i32(<4 x i32> %insert)
  ret <4 x i32> %ret
}

; Test v2i64 insertion into an undef, with an arbitrary index.
define <2 x i64> @f5(ptr %ptr) {
; CHECK-LABEL: f5:
; CHECK: vlbrrepg %v24, 0(%r2)
; CHECK-NEXT: br %r14
  %val = load i64, ptr %ptr
  %swap = call i64 @llvm.bswap.i64(i64 %val)
  %ret = insertelement <2 x i64> undef, i64 %swap, i32 1
  ret <2 x i64> %ret
}

; Test v2i64 insertion into an undef, using a vector bwap.
define <2 x i64> @f6(ptr %ptr) {
; CHECK-LABEL: f6:
; CHECK: vlbrrepg %v24, 0(%r2)
; CHECK-NEXT: br %r14
  %val = load i64, ptr %ptr
  %insert = insertelement <2 x i64> undef, i64 %val, i32 1
  %ret = call <2 x i64> @llvm.bswap.v2i64(<2 x i64> %insert)
  ret <2 x i64> %ret
}

