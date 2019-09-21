; Test vector extraction of byte-swapped value to memory.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z15 | FileCheck %s

declare i16 @llvm.bswap.i16(i16)
declare i32 @llvm.bswap.i32(i32)
declare i64 @llvm.bswap.i64(i64)
declare <8 x i16> @llvm.bswap.v8i16(<8 x i16>)
declare <4 x i32> @llvm.bswap.v4i32(<4 x i32>)
declare <2 x i64> @llvm.bswap.v2i64(<2 x i64>)

; Test v8i16 extraction from the first element.
define void @f1(<8 x i16> %val, i16 *%ptr) {
; CHECK-LABEL: f1:
; CHECK: vstebrh %v24, 0(%r2), 0
; CHECK: br %r14
  %element = extractelement <8 x i16> %val, i32 0
  %swap = call i16 @llvm.bswap.i16(i16 %element)
  store i16 %swap, i16 *%ptr
  ret void
}

; Test v8i16 extraction from the last element.
define void @f2(<8 x i16> %val, i16 *%ptr) {
; CHECK-LABEL: f2:
; CHECK: vstebrh %v24, 0(%r2), 7
; CHECK: br %r14
  %element = extractelement <8 x i16> %val, i32 7
  %swap = call i16 @llvm.bswap.i16(i16 %element)
  store i16 %swap, i16 *%ptr
  ret void
}

; Test v8i16 extraction of an invalid element.  This must compile,
; but we don't care what it does.
define void @f3(<8 x i16> %val, i16 *%ptr) {
; CHECK-LABEL: f3:
; CHECK-NOT: vstebrh %v24, 0(%r2), 8
; CHECK: br %r14
  %element = extractelement <8 x i16> %val, i32 8
  %swap = call i16 @llvm.bswap.i16(i16 %element)
  store i16 %swap, i16 *%ptr
  ret void
}

; Test v8i16 extraction with the highest in-range offset.
define void @f4(<8 x i16> %val, i16 *%base) {
; CHECK-LABEL: f4:
; CHECK: vstebrh %v24, 4094(%r2), 5
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%base, i32 2047
  %element = extractelement <8 x i16> %val, i32 5
  %swap = call i16 @llvm.bswap.i16(i16 %element)
  store i16 %swap, i16 *%ptr
  ret void
}

; Test v8i16 extraction with the first ouf-of-range offset.
define void @f5(<8 x i16> %val, i16 *%base) {
; CHECK-LABEL: f5:
; CHECK: aghi %r2, 4096
; CHECK: vstebrh %v24, 0(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%base, i32 2048
  %element = extractelement <8 x i16> %val, i32 1
  %swap = call i16 @llvm.bswap.i16(i16 %element)
  store i16 %swap, i16 *%ptr
  ret void
}

; Test v8i16 extraction from a variable element.
define void @f6(<8 x i16> %val, i16 *%ptr, i32 %index) {
; CHECK-LABEL: f6:
; CHECK-NOT: vstebrh
; CHECK: br %r14
  %element = extractelement <8 x i16> %val, i32 %index
  %swap = call i16 @llvm.bswap.i16(i16 %element)
  store i16 %swap, i16 *%ptr
  ret void
}

; Test v8i16 extraction using a vector bswap.
define void @f7(<8 x i16> %val, i16 *%ptr) {
; CHECK-LABEL: f7:
; CHECK: vstebrh %v24, 0(%r2), 0
; CHECK: br %r14
  %swap = call <8 x i16> @llvm.bswap.v8i16(<8 x i16> %val)
  %element = extractelement <8 x i16> %swap, i32 0
  store i16 %element, i16 *%ptr
  ret void
}

; Test v4i32 extraction from the first element.
define void @f8(<4 x i32> %val, i32 *%ptr) {
; CHECK-LABEL: f8:
; CHECK: vstebrf %v24, 0(%r2), 0
; CHECK: br %r14
  %element = extractelement <4 x i32> %val, i32 0
  %swap = call i32 @llvm.bswap.i32(i32 %element)
  store i32 %swap, i32 *%ptr
  ret void
}

; Test v4i32 extraction from the last element.
define void @f9(<4 x i32> %val, i32 *%ptr) {
; CHECK-LABEL: f9:
; CHECK: vstebrf %v24, 0(%r2), 3
; CHECK: br %r14
  %element = extractelement <4 x i32> %val, i32 3
  %swap = call i32 @llvm.bswap.i32(i32 %element)
  store i32 %swap, i32 *%ptr
  ret void
}

; Test v4i32 extraction of an invalid element.  This must compile,
; but we don't care what it does.
define void @f10(<4 x i32> %val, i32 *%ptr) {
; CHECK-LABEL: f10:
; CHECK-NOT: vstebrf %v24, 0(%r2), 4
; CHECK: br %r14
  %element = extractelement <4 x i32> %val, i32 4
  %swap = call i32 @llvm.bswap.i32(i32 %element)
  store i32 %swap, i32 *%ptr
  ret void
}

; Test v4i32 extraction with the highest in-range offset.
define void @f11(<4 x i32> %val, i32 *%base) {
; CHECK-LABEL: f11:
; CHECK: vstebrf %v24, 4092(%r2), 2
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%base, i32 1023
  %element = extractelement <4 x i32> %val, i32 2
  %swap = call i32 @llvm.bswap.i32(i32 %element)
  store i32 %swap, i32 *%ptr
  ret void
}

; Test v4i32 extraction with the first ouf-of-range offset.
define void @f12(<4 x i32> %val, i32 *%base) {
; CHECK-LABEL: f12:
; CHECK: aghi %r2, 4096
; CHECK: vstebrf %v24, 0(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%base, i32 1024
  %element = extractelement <4 x i32> %val, i32 1
  %swap = call i32 @llvm.bswap.i32(i32 %element)
  store i32 %swap, i32 *%ptr
  ret void
}

; Test v4i32 extraction from a variable element.
define void @f13(<4 x i32> %val, i32 *%ptr, i32 %index) {
; CHECK-LABEL: f13:
; CHECK-NOT: vstebrf
; CHECK: br %r14
  %element = extractelement <4 x i32> %val, i32 %index
  %swap = call i32 @llvm.bswap.i32(i32 %element)
  store i32 %swap, i32 *%ptr
  ret void
}

; Test v4i32 extraction using a vector bswap.
define void @f14(<4 x i32> %val, i32 *%ptr) {
; CHECK-LABEL: f14:
; CHECK: vstebrf %v24, 0(%r2), 0
; CHECK: br %r14
  %swap = call <4 x i32> @llvm.bswap.v4i32(<4 x i32> %val)
  %element = extractelement <4 x i32> %swap, i32 0
  store i32 %element, i32 *%ptr
  ret void
}

; Test v2i64 extraction from the first element.
define void @f15(<2 x i64> %val, i64 *%ptr) {
; CHECK-LABEL: f15:
; CHECK: vstebrg %v24, 0(%r2), 0
; CHECK: br %r14
  %element = extractelement <2 x i64> %val, i32 0
  %swap = call i64 @llvm.bswap.i64(i64 %element)
  store i64 %swap, i64 *%ptr
  ret void
}

; Test v2i64 extraction from the last element.
define void @f16(<2 x i64> %val, i64 *%ptr) {
; CHECK-LABEL: f16:
; CHECK: vstebrg %v24, 0(%r2), 1
; CHECK: br %r14
  %element = extractelement <2 x i64> %val, i32 1
  %swap = call i64 @llvm.bswap.i64(i64 %element)
  store i64 %swap, i64 *%ptr
  ret void
}

; Test v2i64 extraction of an invalid element.  This must compile,
; but we don't care what it does.
define void @f17(<2 x i64> %val, i64 *%ptr) {
; CHECK-LABEL: f17:
; CHECK-NOT: vstebrg %v24, 0(%r2), 2
; CHECK: br %r14
  %element = extractelement <2 x i64> %val, i32 2
  %swap = call i64 @llvm.bswap.i64(i64 %element)
  store i64 %swap, i64 *%ptr
  ret void
}

; Test v2i64 extraction with the highest in-range offset.
define void @f18(<2 x i64> %val, i64 *%base) {
; CHECK-LABEL: f18:
; CHECK: vstebrg %v24, 4088(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%base, i32 511
  %element = extractelement <2 x i64> %val, i32 1
  %swap = call i64 @llvm.bswap.i64(i64 %element)
  store i64 %swap, i64 *%ptr
  ret void
}

; Test v2i64 extraction with the first ouf-of-range offset.
define void @f19(<2 x i64> %val, i64 *%base) {
; CHECK-LABEL: f19:
; CHECK: aghi %r2, 4096
; CHECK: vstebrg %v24, 0(%r2), 0
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%base, i32 512
  %element = extractelement <2 x i64> %val, i32 0
  %swap = call i64 @llvm.bswap.i64(i64 %element)
  store i64 %swap, i64 *%ptr
  ret void
}

; Test v2i64 extraction from a variable element.
define void @f20(<2 x i64> %val, i64 *%ptr, i32 %index) {
; CHECK-LABEL: f20:
; CHECK-NOT: vstebrg
; CHECK: br %r14
  %element = extractelement <2 x i64> %val, i32 %index
  %swap = call i64 @llvm.bswap.i64(i64 %element)
  store i64 %swap, i64 *%ptr
  ret void
}

; Test v2i64 extraction using a vector bswap.
define void @f21(<2 x i64> %val, i64 *%ptr) {
; CHECK-LABEL: f21:
; CHECK: vstebrg %v24, 0(%r2), 0
; CHECK: br %r14
  %swap = call <2 x i64> @llvm.bswap.v2i64(<2 x i64> %val)
  %element = extractelement <2 x i64> %swap, i32 0
  store i64 %element, i64 *%ptr
  ret void
}

