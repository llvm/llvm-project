; RUN: opt -S -slp-threshold=-6 -slp-vectorizer -instcombine < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; These tests ensure that we do not regress due to PR31243. Note that we set
; the SLP threshold to force vectorization even when not profitable.

; CHECK-LABEL: @PR31243_zext
;
; When computing minimum sizes, if we can prove the sign bit is zero, we can
; zero-extend the roots back to their original sizes.
;
; CHECK: %[[OR:.+]] = or <2 x i8> {{.*}}, <i8 1, i8 1>
; CHECK: %[[E0:.+]] = extractelement <2 x i8> %[[OR]], i32 0
; CHECK: %[[Z0:.+]] = zext i8 %[[E0]] to i64
; CHECK: getelementptr inbounds i8, i8* %ptr, i64 %[[Z0]]
; CHECK: %[[E1:.+]] = extractelement <2 x i8> %[[OR]], i32 1
; CHECK: %[[Z1:.+]] = zext i8 %[[E1]] to i64
; CHECK: getelementptr inbounds i8, i8* %ptr, i64 %[[Z1]]
;
define i8 @PR31243_zext(i8 %v0, i8 %v1, i8 %v2, i8 %v3, i8* %ptr) {
entry:
  %tmp0 = zext i8 %v0 to i32
  %tmp1 = zext i8 %v1 to i32
  %tmp2 = or i32 %tmp0, 1
  %tmp3 = or i32 %tmp1, 1
  %tmp4 = getelementptr inbounds i8, i8* %ptr, i32 %tmp2
  %tmp5 = getelementptr inbounds i8, i8* %ptr, i32 %tmp3
  %tmp6 = load i8, i8* %tmp4
  %tmp7 = load i8, i8* %tmp5
  %tmp8 = add i8 %tmp6, %tmp7
  ret i8 %tmp8
}

; CHECK-LABEL: @PR31243_sext
;
; When computing minimum sizes, if we cannot prove the sign bit is zero, we
; have to include one extra bit for signedness since we will sign-extend the
; roots.
;
; FIXME: This test is suboptimal since the compuation can be performed in i8.
;        In general, we need to add an extra bit to the maximum bit width only
;        if we can't prove that the upper bit of the original type is equal to
;        the upper bit of the proposed smaller type. If these two bits are the
;        same (either zero or one) we know that sign-extending from the smaller
;        type will result in the same value. Since we don't yet perform this
;        optimization, we make the proposed smaller type (i8) larger (i16) to
;        ensure correctness.
;
; CHECK: %[[S0:.+]] = sext <2 x i8> {{.*}} to <2 x i16>
; CHECK: %[[OR:.+]] = or <2 x i16> %[[S0]], <i16 1, i16 1>
; CHECK: %[[E0:.+]] = extractelement <2 x i16> %[[OR]], i32 0
; CHECK: %[[S1:.+]] = sext i16 %[[E0]] to i64
; CHECK: getelementptr inbounds i8, i8* %ptr, i64 %[[S1]]
; CHECK: %[[E1:.+]] = extractelement <2 x i16> %[[OR]], i32 1
; CHECK: %[[S2:.+]] = sext i16 %[[E1]] to i64
; CHECK: getelementptr inbounds i8, i8* %ptr, i64 %[[S2]]
;
define i8 @PR31243_sext(i8 %v0, i8 %v1, i8 %v2, i8 %v3, i8* %ptr) {
entry:
  %tmp0 = sext i8 %v0 to i32
  %tmp1 = sext i8 %v1 to i32
  %tmp2 = or i32 %tmp0, 1
  %tmp3 = or i32 %tmp1, 1
  %tmp4 = getelementptr inbounds i8, i8* %ptr, i32 %tmp2
  %tmp5 = getelementptr inbounds i8, i8* %ptr, i32 %tmp3
  %tmp6 = load i8, i8* %tmp4
  %tmp7 = load i8, i8* %tmp5
  %tmp8 = add i8 %tmp6, %tmp7
  ret i8 %tmp8
}
