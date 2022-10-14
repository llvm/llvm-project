; Test loads of byte-swapped vector elements.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z15 | FileCheck %s

declare <8 x i16> @llvm.bswap.v8i16(<8 x i16>)
declare <4 x i32> @llvm.bswap.v4i32(<4 x i32>)
declare <2 x i64> @llvm.bswap.v2i64(<2 x i64>)

; Test v8i16 loads.
define <8 x i16> @f1(ptr %ptr) {
; CHECK-LABEL: f1:
; CHECK: vlbrh %v24, 0(%r2)
; CHECK: br %r14
  %load = load <8 x i16>, ptr %ptr
  %ret = call <8 x i16> @llvm.bswap.v8i16(<8 x i16> %load)
  ret <8 x i16> %ret
}

; Test v4i32 loads.
define <4 x i32> @f2(ptr %ptr) {
; CHECK-LABEL: f2:
; CHECK: vlbrf %v24, 0(%r2)
; CHECK: br %r14
  %load = load <4 x i32>, ptr %ptr
  %ret = call <4 x i32> @llvm.bswap.v4i32(<4 x i32> %load)
  ret <4 x i32> %ret
}

; Test v2i64 loads.
define <2 x i64> @f3(ptr %ptr) {
; CHECK-LABEL: f3:
; CHECK: vlbrg %v24, 0(%r2)
; CHECK: br %r14
  %load = load <2 x i64>, ptr %ptr
  %ret = call <2 x i64> @llvm.bswap.v2i64(<2 x i64> %load)
  ret <2 x i64> %ret
}

; Test the highest aligned in-range offset.
define <4 x i32> @f4(ptr %base) {
; CHECK-LABEL: f4:
; CHECK: vlbrf %v24, 4080(%r2)
; CHECK: br %r14
  %ptr = getelementptr <4 x i32>, ptr %base, i64 255
  %load = load <4 x i32>, ptr %ptr
  %ret = call <4 x i32> @llvm.bswap.v4i32(<4 x i32> %load)
  ret <4 x i32> %ret
}

; Test the highest unaligned in-range offset.
define <4 x i32> @f5(ptr %base) {
; CHECK-LABEL: f5:
; CHECK: vlbrf %v24, 4095(%r2)
; CHECK: br %r14
  %addr = getelementptr i8, ptr %base, i64 4095
  %load = load <4 x i32>, ptr %addr
  %ret = call <4 x i32> @llvm.bswap.v4i32(<4 x i32> %load)
  ret <4 x i32> %ret
}

; Test the next offset up, which requires separate address logic,
define <4 x i32> @f6(ptr %base) {
; CHECK-LABEL: f6:
; CHECK: aghi %r2, 4096
; CHECK: vlbrf %v24, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr <4 x i32>, ptr %base, i64 256
  %load = load <4 x i32>, ptr %ptr
  %ret = call <4 x i32> @llvm.bswap.v4i32(<4 x i32> %load)
  ret <4 x i32> %ret
}

; Test negative offsets, which also require separate address logic,
define <4 x i32> @f7(ptr %base) {
; CHECK-LABEL: f7:
; CHECK: aghi %r2, -16
; CHECK: vlbrf %v24, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr <4 x i32>, ptr %base, i64 -1
  %load = load <4 x i32>, ptr %ptr
  %ret = call <4 x i32> @llvm.bswap.v4i32(<4 x i32> %load)
  ret <4 x i32> %ret
}

; Check that indexes are allowed.
define <4 x i32> @f8(ptr %base, i64 %index) {
; CHECK-LABEL: f8:
; CHECK: vlbrf %v24, 0(%r3,%r2)
; CHECK: br %r14
  %addr = getelementptr i8, ptr %base, i64 %index
  %load = load <4 x i32>, ptr %addr
  %ret = call <4 x i32> @llvm.bswap.v4i32(<4 x i32> %load)
  ret <4 x i32> %ret
}

