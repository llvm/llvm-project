; Test vector loads.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test v16i8 loads.
define <16 x i8> @f1(ptr %ptr) {
; CHECK-LABEL: f1:
; CHECK: vl %v24, 0(%r2), 3
; CHECK: br %r14
  %ret = load <16 x i8>, ptr %ptr
  ret <16 x i8> %ret
}

; Test v8i16 loads.
define <8 x i16> @f2(ptr %ptr) {
; CHECK-LABEL: f2:
; CHECK: vl %v24, 0(%r2), 3
; CHECK: br %r14
  %ret = load <8 x i16>, ptr %ptr
  ret <8 x i16> %ret
}

; Test v4i32 loads.
define <4 x i32> @f3(ptr %ptr) {
; CHECK-LABEL: f3:
; CHECK: vl %v24, 0(%r2), 3
; CHECK: br %r14
  %ret = load <4 x i32>, ptr %ptr
  ret <4 x i32> %ret
}

; Test v2i64 loads.
define <2 x i64> @f4(ptr %ptr) {
; CHECK-LABEL: f4:
; CHECK: vl %v24, 0(%r2), 3
; CHECK: br %r14
  %ret = load <2 x i64>, ptr %ptr
  ret <2 x i64> %ret
}

; Test v4f32 loads.
define <4 x float> @f5(ptr %ptr) {
; CHECK-LABEL: f5:
; CHECK: vl %v24, 0(%r2), 3
; CHECK: br %r14
  %ret = load <4 x float>, ptr %ptr
  ret <4 x float> %ret
}

; Test v2f64 loads.
define <2 x double> @f6(ptr %ptr) {
; CHECK-LABEL: f6:
; CHECK: vl %v24, 0(%r2), 3
; CHECK: br %r14
  %ret = load <2 x double>, ptr %ptr
  ret <2 x double> %ret
}

; Test the highest aligned in-range offset.
define <16 x i8> @f7(ptr %base) {
; CHECK-LABEL: f7:
; CHECK: vl %v24, 4080(%r2), 3
; CHECK: br %r14
  %ptr = getelementptr <16 x i8>, ptr %base, i64 255
  %ret = load <16 x i8>, ptr %ptr
  ret <16 x i8> %ret
}

; Test the highest unaligned in-range offset.
define <16 x i8> @f8(ptr %base) {
; CHECK-LABEL: f8:
; CHECK: vl %v24, 4095(%r2)
; CHECK: br %r14
  %addr = getelementptr i8, ptr %base, i64 4095
  %ret = load <16 x i8>, ptr %addr, align 1
  ret <16 x i8> %ret
}

; Test the next offset up, which requires separate address logic,
define <16 x i8> @f9(ptr %base) {
; CHECK-LABEL: f9:
; CHECK: aghi %r2, 4096
; CHECK: vl %v24, 0(%r2), 3
; CHECK: br %r14
  %ptr = getelementptr <16 x i8>, ptr %base, i64 256
  %ret = load <16 x i8>, ptr %ptr
  ret <16 x i8> %ret
}

; Test negative offsets, which also require separate address logic,
define <16 x i8> @f10(ptr %base) {
; CHECK-LABEL: f10:
; CHECK: aghi %r2, -16
; CHECK: vl %v24, 0(%r2), 3
; CHECK: br %r14
  %ptr = getelementptr <16 x i8>, ptr %base, i64 -1
  %ret = load <16 x i8>, ptr %ptr
  ret <16 x i8> %ret
}

; Check that indexes are allowed.
define <16 x i8> @f11(ptr %base, i64 %index) {
; CHECK-LABEL: f11:
; CHECK: vl %v24, 0(%r3,%r2)
; CHECK: br %r14
  %addr = getelementptr i8, ptr %base, i64 %index
  %ret = load <16 x i8>, ptr %addr, align 1
  ret <16 x i8> %ret
}

; Test v2i8 loads.
define <2 x i8> @f12(ptr %ptr) {
; CHECK-LABEL: f12:
; CHECK: vlreph %v24, 0(%r2)
; CHECK: br %r14
  %ret = load <2 x i8>, ptr %ptr
  ret <2 x i8> %ret
}

; Test v4i8 loads.
define <4 x i8> @f13(ptr %ptr) {
; CHECK-LABEL: f13:
; CHECK: vlrepf %v24, 0(%r2)
; CHECK: br %r14
  %ret = load <4 x i8>, ptr %ptr
  ret <4 x i8> %ret
}

; Test v8i8 loads.
define <8 x i8> @f14(ptr %ptr) {
; CHECK-LABEL: f14:
; CHECK: vlrepg %v24, 0(%r2)
; CHECK: br %r14
  %ret = load <8 x i8>, ptr %ptr
  ret <8 x i8> %ret
}

; Test v2i16 loads.
define <2 x i16> @f15(ptr %ptr) {
; CHECK-LABEL: f15:
; CHECK: vlrepf %v24, 0(%r2)
; CHECK: br %r14
  %ret = load <2 x i16>, ptr %ptr
  ret <2 x i16> %ret
}

; Test v4i16 loads.
define <4 x i16> @f16(ptr %ptr) {
; CHECK-LABEL: f16:
; CHECK: vlrepg %v24, 0(%r2)
; CHECK: br %r14
  %ret = load <4 x i16>, ptr %ptr
  ret <4 x i16> %ret
}

; Test v2i32 loads.
define <2 x i32> @f17(ptr %ptr) {
; CHECK-LABEL: f17:
; CHECK: vlrepg %v24, 0(%r2)
; CHECK: br %r14
  %ret = load <2 x i32>, ptr %ptr
  ret <2 x i32> %ret
}

; Test v2f32 loads.
define <2 x float> @f18(ptr %ptr) {
; CHECK-LABEL: f18:
; CHECK: vlrepg %v24, 0(%r2)
; CHECK: br %r14
  %ret = load <2 x float>, ptr %ptr
  ret <2 x float> %ret
}

; Test quadword-aligned loads.
define <16 x i8> @f19(ptr %ptr) {
; CHECK-LABEL: f19:
; CHECK: vl %v24, 0(%r2), 4
; CHECK: br %r14
  %ret = load <16 x i8>, ptr %ptr, align 16
  ret <16 x i8> %ret
}

