; Test replications of a byte-swapped scalar memory value.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z15 | FileCheck %s

declare i16 @llvm.bswap.i16(i16)
declare i32 @llvm.bswap.i32(i32)
declare i64 @llvm.bswap.i64(i64)
declare <8 x i16> @llvm.bswap.v8i16(<8 x i16>)
declare <4 x i32> @llvm.bswap.v4i32(<4 x i32>)
declare <2 x i64> @llvm.bswap.v2i64(<2 x i64>)

; Test a v8i16 replicating load with no offset.
define <8 x i16> @f1(ptr %ptr) {
; CHECK-LABEL: f1:
; CHECK: vlbrreph %v24, 0(%r2)
; CHECK: br %r14
  %scalar = load i16, ptr %ptr
  %swap = call i16 @llvm.bswap.i16(i16 %scalar)
  %val = insertelement <8 x i16> undef, i16 %swap, i32 0
  %ret = shufflevector <8 x i16> %val, <8 x i16> undef,
                       <8 x i32> zeroinitializer
  ret <8 x i16> %ret
}

; Test a v8i16 replicating load with the maximum in-range offset.
define <8 x i16> @f2(ptr %base) {
; CHECK-LABEL: f2:
; CHECK: vlbrreph %v24, 4094(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %base, i64 2047
  %scalar = load i16, ptr %ptr
  %swap = call i16 @llvm.bswap.i16(i16 %scalar)
  %val = insertelement <8 x i16> undef, i16 %swap, i32 0
  %ret = shufflevector <8 x i16> %val, <8 x i16> undef,
                       <8 x i32> zeroinitializer
  ret <8 x i16> %ret
}

; Test a v8i16 replicating load with the first out-of-range offset.
define <8 x i16> @f3(ptr %base) {
; CHECK-LABEL: f3:
; CHECK: aghi %r2, 4096
; CHECK: vlbrreph %v24, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %base, i64 2048
  %scalar = load i16, ptr %ptr
  %swap = call i16 @llvm.bswap.i16(i16 %scalar)
  %val = insertelement <8 x i16> undef, i16 %swap, i32 0
  %ret = shufflevector <8 x i16> %val, <8 x i16> undef,
                       <8 x i32> zeroinitializer
  ret <8 x i16> %ret
}

; Test a v8i16 replicating load using a vector bswap.
define <8 x i16> @f4(ptr %ptr) {
; CHECK-LABEL: f4:
; CHECK: vlbrreph %v24, 0(%r2)
; CHECK: br %r14
  %scalar = load i16, ptr %ptr
  %val = insertelement <8 x i16> undef, i16 %scalar, i32 0
  %rep = shufflevector <8 x i16> %val, <8 x i16> undef,
                       <8 x i32> zeroinitializer
  %ret = call <8 x i16> @llvm.bswap.v8i16(<8 x i16> %rep)
  ret <8 x i16> %ret
}

; Test a v4i32 replicating load with no offset.
define <4 x i32> @f5(ptr %ptr) {
; CHECK-LABEL: f5:
; CHECK: vlbrrepf %v24, 0(%r2)
; CHECK: br %r14
  %scalar = load i32, ptr %ptr
  %swap = call i32 @llvm.bswap.i32(i32 %scalar)
  %val = insertelement <4 x i32> undef, i32 %swap, i32 0
  %ret = shufflevector <4 x i32> %val, <4 x i32> undef,
                       <4 x i32> zeroinitializer
  ret <4 x i32> %ret
}

; Test a v4i32 replicating load with the maximum in-range offset.
define <4 x i32> @f6(ptr %base) {
; CHECK-LABEL: f6:
; CHECK: vlbrrepf %v24, 4092(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %base, i64 1023
  %scalar = load i32, ptr %ptr
  %swap = call i32 @llvm.bswap.i32(i32 %scalar)
  %val = insertelement <4 x i32> undef, i32 %swap, i32 0
  %ret = shufflevector <4 x i32> %val, <4 x i32> undef,
                       <4 x i32> zeroinitializer
  ret <4 x i32> %ret
}

; Test a v4i32 replicating load with the first out-of-range offset.
define <4 x i32> @f7(ptr %base) {
; CHECK-LABEL: f7:
; CHECK: aghi %r2, 4096
; CHECK: vlbrrepf %v24, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %base, i64 1024
  %scalar = load i32, ptr %ptr
  %swap = call i32 @llvm.bswap.i32(i32 %scalar)
  %val = insertelement <4 x i32> undef, i32 %swap, i32 0
  %ret = shufflevector <4 x i32> %val, <4 x i32> undef,
                       <4 x i32> zeroinitializer
  ret <4 x i32> %ret
}

; Test a v4i32 replicating load using a vector bswap.
define <4 x i32> @f8(ptr %ptr) {
; CHECK-LABEL: f8:
; CHECK: vlbrrepf %v24, 0(%r2)
; CHECK: br %r14
  %scalar = load i32, ptr %ptr
  %val = insertelement <4 x i32> undef, i32 %scalar, i32 0
  %rep = shufflevector <4 x i32> %val, <4 x i32> undef,
                       <4 x i32> zeroinitializer
  %ret = call <4 x i32> @llvm.bswap.v4i32(<4 x i32> %rep)
  ret <4 x i32> %ret
}

; Test a v2i64 replicating load with no offset.
define <2 x i64> @f9(ptr %ptr) {
; CHECK-LABEL: f9:
; CHECK: vlbrrepg %v24, 0(%r2)
; CHECK: br %r14
  %scalar = load i64, ptr %ptr
  %swap = call i64 @llvm.bswap.i64(i64 %scalar)
  %val = insertelement <2 x i64> undef, i64 %swap, i32 0
  %ret = shufflevector <2 x i64> %val, <2 x i64> undef,
                       <2 x i32> zeroinitializer
  ret <2 x i64> %ret
}

; Test a v2i64 replicating load with the maximum in-range offset.
define <2 x i64> @f10(ptr %base) {
; CHECK-LABEL: f10:
; CHECK: vlbrrepg %v24, 4088(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %base, i32 511
  %scalar = load i64, ptr %ptr
  %swap = call i64 @llvm.bswap.i64(i64 %scalar)
  %val = insertelement <2 x i64> undef, i64 %swap, i32 0
  %ret = shufflevector <2 x i64> %val, <2 x i64> undef,
                       <2 x i32> zeroinitializer
  ret <2 x i64> %ret
}

; Test a v2i64 replicating load with the first out-of-range offset.
define <2 x i64> @f11(ptr %base) {
; CHECK-LABEL: f11:
; CHECK: aghi %r2, 4096
; CHECK: vlbrrepg %v24, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %base, i32 512
  %scalar = load i64, ptr %ptr
  %swap = call i64 @llvm.bswap.i64(i64 %scalar)
  %val = insertelement <2 x i64> undef, i64 %swap, i32 0
  %ret = shufflevector <2 x i64> %val, <2 x i64> undef,
                       <2 x i32> zeroinitializer
  ret <2 x i64> %ret
}

; Test a v2i64 replicating load using a vector bswap.
define <2 x i64> @f12(ptr %ptr) {
; CHECK-LABEL: f12:
; CHECK: vlbrrepg %v24, 0(%r2)
; CHECK: br %r14
  %scalar = load i64, ptr %ptr
  %val = insertelement <2 x i64> undef, i64 %scalar, i32 0
  %rep = shufflevector <2 x i64> %val, <2 x i64> undef,
                       <2 x i32> zeroinitializer
  %ret = call <2 x i64> @llvm.bswap.v2i64(<2 x i64> %rep)
  ret <2 x i64> %ret
}

; Test a v8i16 replicating load with an index.
define <8 x i16> @f13(ptr %base, i64 %index) {
; CHECK-LABEL: f13:
; CHECK: sllg [[REG:%r[1-5]]], %r3, 1
; CHECK: vlbrreph %v24, 2046([[REG]],%r2)
; CHECK: br %r14
  %ptr1 = getelementptr i16, ptr %base, i64 %index
  %ptr = getelementptr i16, ptr %ptr1, i64 1023
  %scalar = load i16, ptr %ptr
  %swap = call i16 @llvm.bswap.i16(i16 %scalar)
  %val = insertelement <8 x i16> undef, i16 %swap, i32 0
  %ret = shufflevector <8 x i16> %val, <8 x i16> undef,
                       <8 x i32> zeroinitializer
  ret <8 x i16> %ret
}

