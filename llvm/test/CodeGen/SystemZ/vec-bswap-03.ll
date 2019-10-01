; Test vector insertion of byte-swapped memory values.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z15 | FileCheck %s

declare i16 @llvm.bswap.i16(i16)
declare i32 @llvm.bswap.i32(i32)
declare i64 @llvm.bswap.i64(i64)
declare <8 x i16> @llvm.bswap.v8i16(<8 x i16>)
declare <4 x i32> @llvm.bswap.v4i32(<4 x i32>)
declare <2 x i64> @llvm.bswap.v2i64(<2 x i64>)

; Test v8i16 insertion into the first element.
define <8 x i16> @f1(<8 x i16> %val, i16 *%ptr) {
; CHECK-LABEL: f1:
; CHECK: vlebrh %v24, 0(%r2), 0
; CHECK: br %r14
  %element = load i16, i16 *%ptr
  %swap = call i16 @llvm.bswap.i16(i16 %element)
  %ret = insertelement <8 x i16> %val, i16 %swap, i32 0
  ret <8 x i16> %ret
}

; Test v8i16 insertion into the last element.
define <8 x i16> @f2(<8 x i16> %val, i16 *%ptr) {
; CHECK-LABEL: f2:
; CHECK: vlebrh %v24, 0(%r2), 7
; CHECK: br %r14
  %element = load i16, i16 *%ptr
  %swap = call i16 @llvm.bswap.i16(i16 %element)
  %ret = insertelement <8 x i16> %val, i16 %swap, i32 7
  ret <8 x i16> %ret
}

; Test v8i16 insertion with the highest in-range offset.
define <8 x i16> @f3(<8 x i16> %val, i16 *%base) {
; CHECK-LABEL: f3:
; CHECK: vlebrh %v24, 4094(%r2), 5
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%base, i32 2047
  %element = load i16, i16 *%ptr
  %swap = call i16 @llvm.bswap.i16(i16 %element)
  %ret = insertelement <8 x i16> %val, i16 %swap, i32 5
  ret <8 x i16> %ret
}

; Test v8i16 insertion with the first ouf-of-range offset.
define <8 x i16> @f4(<8 x i16> %val, i16 *%base) {
; CHECK-LABEL: f4:
; CHECK: aghi %r2, 4096
; CHECK: vlebrh %v24, 0(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%base, i32 2048
  %element = load i16, i16 *%ptr
  %swap = call i16 @llvm.bswap.i16(i16 %element)
  %ret = insertelement <8 x i16> %val, i16 %swap, i32 1
  ret <8 x i16> %ret
}

; Test v8i16 insertion into a variable element.
define <8 x i16> @f5(<8 x i16> %val, i16 *%ptr, i32 %index) {
; CHECK-LABEL: f5:
; CHECK-NOT: vlebrh
; CHECK: br %r14
  %element = load i16, i16 *%ptr
  %swap = call i16 @llvm.bswap.i16(i16 %element)
  %ret = insertelement <8 x i16> %val, i16 %swap, i32 %index
  ret <8 x i16> %ret
}

; Test v8i16 insertion using a pair of vector bswaps.
define <8 x i16> @f6(<8 x i16> %val, i16 *%ptr) {
; CHECK-LABEL: f6:
; CHECK: vlebrh %v24, 0(%r2), 0
; CHECK: br %r14
  %element = load i16, i16 *%ptr
  %swapval = call <8 x i16> @llvm.bswap.v8i16(<8 x i16> %val)
  %insert = insertelement <8 x i16> %swapval, i16 %element, i32 0
  %ret = call <8 x i16> @llvm.bswap.v8i16(<8 x i16> %insert)
  ret <8 x i16> %ret
}

; Test v4i32 insertion into the first element.
define <4 x i32> @f7(<4 x i32> %val, i32 *%ptr) {
; CHECK-LABEL: f7:
; CHECK: vlebrf %v24, 0(%r2), 0
; CHECK: br %r14
  %element = load i32, i32 *%ptr
  %swap = call i32 @llvm.bswap.i32(i32 %element)
  %ret = insertelement <4 x i32> %val, i32 %swap, i32 0
  ret <4 x i32> %ret
}

; Test v4i32 insertion into the last element.
define <4 x i32> @f8(<4 x i32> %val, i32 *%ptr) {
; CHECK-LABEL: f8:
; CHECK: vlebrf %v24, 0(%r2), 3
; CHECK: br %r14
  %element = load i32, i32 *%ptr
  %swap = call i32 @llvm.bswap.i32(i32 %element)
  %ret = insertelement <4 x i32> %val, i32 %swap, i32 3
  ret <4 x i32> %ret
}

; Test v4i32 insertion with the highest in-range offset.
define <4 x i32> @f9(<4 x i32> %val, i32 *%base) {
; CHECK-LABEL: f9:
; CHECK: vlebrf %v24, 4092(%r2), 2
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%base, i32 1023
  %element = load i32, i32 *%ptr
  %swap = call i32 @llvm.bswap.i32(i32 %element)
  %ret = insertelement <4 x i32> %val, i32 %swap, i32 2
  ret <4 x i32> %ret
}

; Test v4i32 insertion with the first ouf-of-range offset.
define <4 x i32> @f10(<4 x i32> %val, i32 *%base) {
; CHECK-LABEL: f10:
; CHECK: aghi %r2, 4096
; CHECK: vlebrf %v24, 0(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%base, i32 1024
  %element = load i32, i32 *%ptr
  %swap = call i32 @llvm.bswap.i32(i32 %element)
  %ret = insertelement <4 x i32> %val, i32 %swap, i32 1
  ret <4 x i32> %ret
}

; Test v4i32 insertion into a variable element.
define <4 x i32> @f11(<4 x i32> %val, i32 *%ptr, i32 %index) {
; CHECK-LABEL: f11:
; CHECK-NOT: vlebrf
; CHECK: br %r14
  %element = load i32, i32 *%ptr
  %swap = call i32 @llvm.bswap.i32(i32 %element)
  %ret = insertelement <4 x i32> %val, i32 %swap, i32 %index
  ret <4 x i32> %ret
}

; Test v4i32 insertion using a pair of vector bswaps.
define <4 x i32> @f12(<4 x i32> %val, i32 *%ptr) {
; CHECK-LABEL: f12:
; CHECK: vlebrf %v24, 0(%r2), 0
; CHECK: br %r14
  %element = load i32, i32 *%ptr
  %swapval = call <4 x i32> @llvm.bswap.v4i32(<4 x i32> %val)
  %insert = insertelement <4 x i32> %swapval, i32 %element, i32 0
  %ret = call <4 x i32> @llvm.bswap.v4i32(<4 x i32> %insert)
  ret <4 x i32> %ret
}

; Test v2i64 insertion into the first element.
define <2 x i64> @f13(<2 x i64> %val, i64 *%ptr) {
; CHECK-LABEL: f13:
; CHECK: vlebrg %v24, 0(%r2), 0
; CHECK: br %r14
  %element = load i64, i64 *%ptr
  %swap = call i64 @llvm.bswap.i64(i64 %element)
  %ret = insertelement <2 x i64> %val, i64 %swap, i32 0
  ret <2 x i64> %ret
}

; Test v2i64 insertion into the last element.
define <2 x i64> @f14(<2 x i64> %val, i64 *%ptr) {
; CHECK-LABEL: f14:
; CHECK: vlebrg %v24, 0(%r2), 1
; CHECK: br %r14
  %element = load i64, i64 *%ptr
  %swap = call i64 @llvm.bswap.i64(i64 %element)
  %ret = insertelement <2 x i64> %val, i64 %swap, i32 1
  ret <2 x i64> %ret
}

; Test v2i64 insertion with the highest in-range offset.
define <2 x i64> @f15(<2 x i64> %val, i64 *%base) {
; CHECK-LABEL: f15:
; CHECK: vlebrg %v24, 4088(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%base, i32 511
  %element = load i64, i64 *%ptr
  %swap = call i64 @llvm.bswap.i64(i64 %element)
  %ret = insertelement <2 x i64> %val, i64 %swap, i32 1
  ret <2 x i64> %ret
}

; Test v2i64 insertion with the first ouf-of-range offset.
define <2 x i64> @f16(<2 x i64> %val, i64 *%base) {
; CHECK-LABEL: f16:
; CHECK: aghi %r2, 4096
; CHECK: vlebrg %v24, 0(%r2), 0
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%base, i32 512
  %element = load i64, i64 *%ptr
  %swap = call i64 @llvm.bswap.i64(i64 %element)
  %ret = insertelement <2 x i64> %val, i64 %swap, i32 0
  ret <2 x i64> %ret
}

; Test v2i64 insertion into a variable element.
define <2 x i64> @f17(<2 x i64> %val, i64 *%ptr, i32 %index) {
; CHECK-LABEL: f17:
; CHECK-NOT: vlebrg
; CHECK: br %r14
  %element = load i64, i64 *%ptr
  %swap = call i64 @llvm.bswap.i64(i64 %element)
  %ret = insertelement <2 x i64> %val, i64 %swap, i32 %index
  ret <2 x i64> %ret
}

; Test v2i64 insertion using a pair of vector bswaps.
define <2 x i64> @f18(<2 x i64> %val, i64 *%ptr) {
; CHECK-LABEL: f18:
; CHECK: vlebrg %v24, 0(%r2), 0
; CHECK: br %r14
  %element = load i64, i64 *%ptr
  %swapval = call <2 x i64> @llvm.bswap.v2i64(<2 x i64> %val)
  %insert = insertelement <2 x i64> %swapval, i64 %element, i32 0
  %ret = call <2 x i64> @llvm.bswap.v2i64(<2 x i64> %insert)
  ret <2 x i64> %ret
}
