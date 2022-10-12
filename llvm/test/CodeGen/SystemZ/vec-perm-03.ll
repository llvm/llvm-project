; Test replications of a scalar memory value, represented as splats.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a v16i8 replicating load with no offset.
define <16 x i8> @f1(ptr %ptr) {
; CHECK-LABEL: f1:
; CHECK: vlrepb %v24, 0(%r2)
; CHECK: br %r14
  %scalar = load i8, ptr %ptr
  %val = insertelement <16 x i8> undef, i8 %scalar, i32 0
  %ret = shufflevector <16 x i8> %val, <16 x i8> undef,
                       <16 x i32> zeroinitializer
  ret <16 x i8> %ret
}

; Test a v16i8 replicating load with the maximum in-range offset.
define <16 x i8> @f2(ptr %base) {
; CHECK-LABEL: f2:
; CHECK: vlrepb %v24, 4095(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %base, i64 4095
  %scalar = load i8, ptr %ptr
  %val = insertelement <16 x i8> undef, i8 %scalar, i32 0
  %ret = shufflevector <16 x i8> %val, <16 x i8> undef,
                       <16 x i32> zeroinitializer
  ret <16 x i8> %ret
}

; Test a v16i8 replicating load with the first out-of-range offset.
define <16 x i8> @f3(ptr %base) {
; CHECK-LABEL: f3:
; CHECK: aghi %r2, 4096
; CHECK: vlrepb %v24, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %base, i64 4096
  %scalar = load i8, ptr %ptr
  %val = insertelement <16 x i8> undef, i8 %scalar, i32 0
  %ret = shufflevector <16 x i8> %val, <16 x i8> undef,
                       <16 x i32> zeroinitializer
  ret <16 x i8> %ret
}

; Test a v8i16 replicating load with no offset.
define <8 x i16> @f4(ptr %ptr) {
; CHECK-LABEL: f4:
; CHECK: vlreph %v24, 0(%r2)
; CHECK: br %r14
  %scalar = load i16, ptr %ptr
  %val = insertelement <8 x i16> undef, i16 %scalar, i32 0
  %ret = shufflevector <8 x i16> %val, <8 x i16> undef,
                       <8 x i32> zeroinitializer
  ret <8 x i16> %ret
}

; Test a v8i16 replicating load with the maximum in-range offset.
define <8 x i16> @f5(ptr %base) {
; CHECK-LABEL: f5:
; CHECK: vlreph %v24, 4094(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %base, i64 2047
  %scalar = load i16, ptr %ptr
  %val = insertelement <8 x i16> undef, i16 %scalar, i32 0
  %ret = shufflevector <8 x i16> %val, <8 x i16> undef,
                       <8 x i32> zeroinitializer
  ret <8 x i16> %ret
}

; Test a v8i16 replicating load with the first out-of-range offset.
define <8 x i16> @f6(ptr %base) {
; CHECK-LABEL: f6:
; CHECK: aghi %r2, 4096
; CHECK: vlreph %v24, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %base, i64 2048
  %scalar = load i16, ptr %ptr
  %val = insertelement <8 x i16> undef, i16 %scalar, i32 0
  %ret = shufflevector <8 x i16> %val, <8 x i16> undef,
                       <8 x i32> zeroinitializer
  ret <8 x i16> %ret
}

; Test a v4i32 replicating load with no offset.
define <4 x i32> @f7(ptr %ptr) {
; CHECK-LABEL: f7:
; CHECK: vlrepf %v24, 0(%r2)
; CHECK: br %r14
  %scalar = load i32, ptr %ptr
  %val = insertelement <4 x i32> undef, i32 %scalar, i32 0
  %ret = shufflevector <4 x i32> %val, <4 x i32> undef,
                       <4 x i32> zeroinitializer
  ret <4 x i32> %ret
}

; Test a v4i32 replicating load with the maximum in-range offset.
define <4 x i32> @f8(ptr %base) {
; CHECK-LABEL: f8:
; CHECK: vlrepf %v24, 4092(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %base, i64 1023
  %scalar = load i32, ptr %ptr
  %val = insertelement <4 x i32> undef, i32 %scalar, i32 0
  %ret = shufflevector <4 x i32> %val, <4 x i32> undef,
                       <4 x i32> zeroinitializer
  ret <4 x i32> %ret
}

; Test a v4i32 replicating load with the first out-of-range offset.
define <4 x i32> @f9(ptr %base) {
; CHECK-LABEL: f9:
; CHECK: aghi %r2, 4096
; CHECK: vlrepf %v24, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %base, i64 1024
  %scalar = load i32, ptr %ptr
  %val = insertelement <4 x i32> undef, i32 %scalar, i32 0
  %ret = shufflevector <4 x i32> %val, <4 x i32> undef,
                       <4 x i32> zeroinitializer
  ret <4 x i32> %ret
}

; Test a v2i64 replicating load with no offset.
define <2 x i64> @f10(ptr %ptr) {
; CHECK-LABEL: f10:
; CHECK: vlrepg %v24, 0(%r2)
; CHECK: br %r14
  %scalar = load i64, ptr %ptr
  %val = insertelement <2 x i64> undef, i64 %scalar, i32 0
  %ret = shufflevector <2 x i64> %val, <2 x i64> undef,
                       <2 x i32> zeroinitializer
  ret <2 x i64> %ret
}

; Test a v2i64 replicating load with the maximum in-range offset.
define <2 x i64> @f11(ptr %base) {
; CHECK-LABEL: f11:
; CHECK: vlrepg %v24, 4088(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %base, i32 511
  %scalar = load i64, ptr %ptr
  %val = insertelement <2 x i64> undef, i64 %scalar, i32 0
  %ret = shufflevector <2 x i64> %val, <2 x i64> undef,
                       <2 x i32> zeroinitializer
  ret <2 x i64> %ret
}

; Test a v2i64 replicating load with the first out-of-range offset.
define <2 x i64> @f12(ptr %base) {
; CHECK-LABEL: f12:
; CHECK: aghi %r2, 4096
; CHECK: vlrepg %v24, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %base, i32 512
  %scalar = load i64, ptr %ptr
  %val = insertelement <2 x i64> undef, i64 %scalar, i32 0
  %ret = shufflevector <2 x i64> %val, <2 x i64> undef,
                       <2 x i32> zeroinitializer
  ret <2 x i64> %ret
}

; Test a v4f32 replicating load with no offset.
define <4 x float> @f13(ptr %ptr) {
; CHECK-LABEL: f13:
; CHECK: vlrepf %v24, 0(%r2)
; CHECK: br %r14
  %scalar = load float, ptr %ptr
  %val = insertelement <4 x float> undef, float %scalar, i32 0
  %ret = shufflevector <4 x float> %val, <4 x float> undef,
                       <4 x i32> zeroinitializer
  ret <4 x float> %ret
}

; Test a v4f32 replicating load with the maximum in-range offset.
define <4 x float> @f14(ptr %base) {
; CHECK-LABEL: f14:
; CHECK: vlrepf %v24, 4092(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, ptr %base, i64 1023
  %scalar = load float, ptr %ptr
  %val = insertelement <4 x float> undef, float %scalar, i32 0
  %ret = shufflevector <4 x float> %val, <4 x float> undef,
                       <4 x i32> zeroinitializer
  ret <4 x float> %ret
}

; Test a v4f32 replicating load with the first out-of-range offset.
define <4 x float> @f15(ptr %base) {
; CHECK-LABEL: f15:
; CHECK: aghi %r2, 4096
; CHECK: vlrepf %v24, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, ptr %base, i64 1024
  %scalar = load float, ptr %ptr
  %val = insertelement <4 x float> undef, float %scalar, i32 0
  %ret = shufflevector <4 x float> %val, <4 x float> undef,
                       <4 x i32> zeroinitializer
  ret <4 x float> %ret
}

; Test a v2f64 replicating load with no offset.
define <2 x double> @f16(ptr %ptr) {
; CHECK-LABEL: f16:
; CHECK: vlrepg %v24, 0(%r2)
; CHECK: br %r14
  %scalar = load double, ptr %ptr
  %val = insertelement <2 x double> undef, double %scalar, i32 0
  %ret = shufflevector <2 x double> %val, <2 x double> undef,
                       <2 x i32> zeroinitializer
  ret <2 x double> %ret
}

; Test a v2f64 replicating load with the maximum in-range offset.
define <2 x double> @f17(ptr %base) {
; CHECK-LABEL: f17:
; CHECK: vlrepg %v24, 4088(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, ptr %base, i32 511
  %scalar = load double, ptr %ptr
  %val = insertelement <2 x double> undef, double %scalar, i32 0
  %ret = shufflevector <2 x double> %val, <2 x double> undef,
                       <2 x i32> zeroinitializer
  ret <2 x double> %ret
}

; Test a v2f64 replicating load with the first out-of-range offset.
define <2 x double> @f18(ptr %base) {
; CHECK-LABEL: f18:
; CHECK: aghi %r2, 4096
; CHECK: vlrepg %v24, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, ptr %base, i32 512
  %scalar = load double, ptr %ptr
  %val = insertelement <2 x double> undef, double %scalar, i32 0
  %ret = shufflevector <2 x double> %val, <2 x double> undef,
                       <2 x i32> zeroinitializer
  ret <2 x double> %ret
}

; Test a v16i8 replicating load with an index.
define <16 x i8> @f19(ptr %base, i64 %index) {
; CHECK-LABEL: f19:
; CHECK: vlrepb %v24, 1023(%r3,%r2)
; CHECK: br %r14
  %ptr1 = getelementptr i8, ptr %base, i64 %index
  %ptr = getelementptr i8, ptr %ptr1, i64 1023
  %scalar = load i8, ptr %ptr
  %val = insertelement <16 x i8> undef, i8 %scalar, i32 0
  %ret = shufflevector <16 x i8> %val, <16 x i8> undef,
                       <16 x i32> zeroinitializer
  ret <16 x i8> %ret
}
