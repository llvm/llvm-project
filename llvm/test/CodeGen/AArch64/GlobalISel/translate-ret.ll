; RUN: llc -mtriple=arm64-apple-ios %s -o - -global-isel -global-isel-abort=1 -stop-after=irtranslator | FileCheck %s

define i128 @func_i128(i128* %ptr) {
; CHECK-LABEL: name: func_i128
; CHECK: [[PTR:%.*]]:_(p0) = COPY $x0
; CHECK: [[VAL:%.*]]:_(s128) = G_LOAD [[PTR]]
; CHECK: [[LO:%.*]]:_(s64), [[HI:%.*]]:_(s64) = G_UNMERGE_VALUES [[VAL]]
; CHECK: $x0 = COPY [[LO]]
; CHECK: $x1 = COPY [[HI]]
; CHECK: RET_ReallyLR

  %val = load i128, i128* %ptr
  ret i128 %val
}

define <8 x float> @func_v8f32(<8 x float>* %ptr) {
; CHECK-LABEL: name: func_v8f32
; CHECK: [[PTR:%.*]]:_(p0) = COPY $x0
; CHECK: [[VAL:%.*]]:_(<8 x s32>) = G_LOAD [[PTR]]
; CHECK: [[LO:%.*]]:_(<4 x s32>), [[HI:%.*]]:_(<4 x s32>) = G_UNMERGE_VALUES [[VAL]]
; CHECK: $q0 = COPY [[LO]]
; CHECK: $q1 = COPY [[HI]]
; CHECK: RET_ReallyLR

  %val = load <8 x float>, <8 x float>* %ptr
  ret <8 x float> %val
}

; A bit weird, but s0-s5 is what SDAG does too.
define <6 x float> @func_v6f32(<6 x float>* %ptr) {
; CHECK-LABEL: name: func_v6f32
; CHECK: [[PTR:%.*]]:_(p0) = COPY $x0
; CHECK: [[VAL:%.*]]:_(<6 x s32>) = G_LOAD [[PTR]]
; CHECK: [[V1:%.*]]:_(s32), [[V2:%.*]]:_(s32), [[V3:%.*]]:_(s32), [[V4:%.*]]:_(s32), [[V5:%.*]]:_(s32), [[V6:%.*]]:_(s32) = G_UNMERGE_VALUES [[VAL]]
; CHECK: $s0 = COPY [[V1]]
; CHECK: $s1 = COPY [[V2]]
; CHECK: $s2 = COPY [[V3]]
; CHECK: $s3 = COPY [[V4]]
; CHECK: $s4 = COPY [[V5]]
; CHECK: $s5 = COPY [[V6]]
; CHECK: RET_ReallyLR

  %val = load <6 x float>, <6 x float>* %ptr
  ret <6 x float> %val
}

define i128 @ABIi128(i128 %arg1) {
; CHECK-LABEL: name: ABIi128
; CHECK: [[LO:%.*]]:_(s64) = COPY $x0
; CHECK: [[HI:%.*]]:_(s64) = COPY $x1
; CHECK: [[IN:%.*]]:_(s128) = G_MERGE_VALUES [[LO]](s64), [[HI]](s64)
; CHECK: [[IN_FP:%.*]]:_(s128) = G_FPTOUI [[IN]](s128)
; CHECK: [[LO_OUT:%.*]]:_(s64), [[HI_OUT:%.*]]:_(s64) = G_UNMERGE_VALUES [[IN_FP]](s128)
; CHECK: $x0 = COPY [[LO_OUT]]
; CHECK: $x1 = COPY [[HI_OUT]]
  %farg1 =       bitcast i128 %arg1 to fp128
  %res = fptoui fp128 %farg1 to i128
  ret i128 %res
}
