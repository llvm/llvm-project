; Test vector insertions of byte-swapped memory values into 0.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z15 | FileCheck %s

declare i16 @llvm.bswap.i16(i16)
declare i32 @llvm.bswap.i32(i32)
declare i64 @llvm.bswap.i64(i64)
declare <8 x i16> @llvm.bswap.v8i16(<8 x i16>)
declare <4 x i32> @llvm.bswap.v4i32(<4 x i32>)
declare <2 x i64> @llvm.bswap.v2i64(<2 x i64>)

; Test VLLEBRZH.
define <8 x i16> @f1(ptr %ptr) {
; CHECK-LABEL: f1:
; CHECK: vllebrzh %v24, 0(%r2)
; CHECK: br %r14
  %val = load i16, ptr %ptr
  %swap = call i16 @llvm.bswap.i16(i16 %val)
  %ret = insertelement <8 x i16> zeroinitializer, i16 %swap, i32 3
  ret <8 x i16> %ret
}

; Test VLLEBRZH using a vector bswap.
define <8 x i16> @f2(ptr %ptr) {
; CHECK-LABEL: f2:
; CHECK: vllebrzh %v24, 0(%r2)
; CHECK: br %r14
  %val = load i16, ptr %ptr
  %insert = insertelement <8 x i16> zeroinitializer, i16 %val, i32 3
  %ret = call <8 x i16> @llvm.bswap.v8i16(<8 x i16> %insert)
  ret <8 x i16> %ret
}

; Test VLLEBRZF.
define <4 x i32> @f3(ptr %ptr) {
; CHECK-LABEL: f3:
; CHECK: vllebrzf %v24, 0(%r2)
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %swap = call i32 @llvm.bswap.i32(i32 %val)
  %ret = insertelement <4 x i32> zeroinitializer, i32 %swap, i32 1
  ret <4 x i32> %ret
}

; Test VLLEBRZF using a vector bswap.
define <4 x i32> @f4(ptr %ptr) {
; CHECK-LABEL: f4:
; CHECK: vllebrzf %v24, 0(%r2)
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %insert = insertelement <4 x i32> zeroinitializer, i32 %val, i32 1
  %ret = call <4 x i32> @llvm.bswap.v4i32(<4 x i32> %insert)
  ret <4 x i32> %ret
}

; Test VLLEBRZG.
define <2 x i64> @f5(ptr %ptr) {
; CHECK-LABEL: f5:
; CHECK: vllebrzg %v24, 0(%r2)
; CHECK: br %r14
  %val = load i64, ptr %ptr
  %swap = call i64 @llvm.bswap.i64(i64 %val)
  %ret = insertelement <2 x i64> zeroinitializer, i64 %swap, i32 0
  ret <2 x i64> %ret
}

; Test VLLEBRZG using a vector bswap.
define <2 x i64> @f6(ptr %ptr) {
; CHECK-LABEL: f6:
; CHECK: vllebrzg %v24, 0(%r2)
; CHECK: br %r14
  %val = load i64, ptr %ptr
  %insert = insertelement <2 x i64> zeroinitializer, i64 %val, i32 0
  %ret = call <2 x i64> @llvm.bswap.v2i64(<2 x i64> %insert)
  ret <2 x i64> %ret
}

; Test VLLEBRZE.
define <4 x i32> @f7(ptr %ptr) {
; CHECK-LABEL: f7:
; CHECK: vllebrze %v24, 0(%r2)
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %swap = call i32 @llvm.bswap.i32(i32 %val)
  %ret = insertelement <4 x i32> zeroinitializer, i32 %swap, i32 0
  ret <4 x i32> %ret
}

; Test VLLEBRZE using a vector bswap.
define <4 x i32> @f8(ptr %ptr) {
; CHECK-LABEL: f8:
; CHECK: vllebrze %v24, 0(%r2)
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %insert = insertelement <4 x i32> zeroinitializer, i32 %val, i32 0
  %ret = call <4 x i32> @llvm.bswap.v4i32(<4 x i32> %insert)
  ret <4 x i32> %ret
}

; Test VLLEBRZH with the highest in-range offset.
define <8 x i16> @f9(ptr %base) {
; CHECK-LABEL: f9:
; CHECK: vllebrzh %v24, 4094(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %base, i64 2047
  %val = load i16, ptr %ptr
  %swap = call i16 @llvm.bswap.i16(i16 %val)
  %ret = insertelement <8 x i16> zeroinitializer, i16 %swap, i32 3
  ret <8 x i16> %ret
}

; Test VLLEBRZH with the next highest offset.
define <8 x i16> @f10(ptr %base) {
; CHECK-LABEL: f10:
; CHECK-NOT: vllebrzh %v24, 4096(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %base, i64 2048
  %val = load i16, ptr %ptr
  %swap = call i16 @llvm.bswap.i16(i16 %val)
  %ret = insertelement <8 x i16> zeroinitializer, i16 %swap, i32 3
  ret <8 x i16> %ret
}

; Test that VLLEBRZH allows an index.
define <8 x i16> @f11(ptr %base, i64 %index) {
; CHECK-LABEL: f11:
; CHECK: sllg [[REG:%r[1-5]]], %r3, 1
; CHECK: vllebrzh %v24, 0([[REG]],%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %base, i64 %index
  %val = load i16, ptr %ptr
  %swap = call i16 @llvm.bswap.i16(i16 %val)
  %ret = insertelement <8 x i16> zeroinitializer, i16 %swap, i32 3
  ret <8 x i16> %ret
}

