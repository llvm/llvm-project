; RUN: opt < %s -interleaved-access -S | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

define void @extract_user_basic(<8 x i32>* %ptr, i1 %c) {
; CHECK-LABEL: @extract_user_basic(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = bitcast <8 x i32>* %ptr to <4 x i32>*
; CHECK-NEXT:    [[LDN:%.*]] = call { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2.v4i32.p0v4i32(<4 x i32>* [[TMP0]])
; CHECK-NEXT:    [[TMP1:%.*]] = extractvalue { <4 x i32>, <4 x i32> } [[LDN]], 0
; CHECK-NEXT:    br i1 %c, label %if.then, label %if.merge
; CHECK:       if.then:
; CHECK-NEXT:    [[TMP2:%.*]] = extractelement <4 x i32> [[TMP1]], i64 1
; CHECK-NEXT:    br label %if.merge
; CHECK:       if.merge:
; CHECK-NEXT:    ret void
;
entry:
  %interleaved.vec = load <8 x i32>, <8 x i32>* %ptr, align 8
  %v0 = shufflevector <8 x i32> %interleaved.vec, <8 x i32> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  br i1 %c, label %if.then, label %if.merge

if.then:
  %e0 = extractelement <8 x i32> %interleaved.vec, i32 2
  br label %if.merge

if.merge:
  ret void
}

define void @extract_user_multi(<8 x i32>* %ptr, i1 %c) {
; CHECK-LABEL: @extract_user_multi(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = bitcast <8 x i32>* %ptr to <4 x i32>*
; CHECK-NEXT:    [[LDN:%.*]] = call { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2.v4i32.p0v4i32(<4 x i32>* [[TMP0]])
; CHECK-NEXT:    [[TMP1:%.*]] = extractvalue { <4 x i32>, <4 x i32> } [[LDN]], 0
; CHECK-NEXT:    br i1 %c, label %if.then, label %if.merge
; CHECK:       if.then:
; CHECK-NEXT:    [[TMP2:%.*]] = extractelement <4 x i32> [[TMP1]], i64 0
; CHECK-NEXT:    br label %if.merge
; CHECK:       if.merge:
; CHECK-NEXT:    [[TMP3:%.*]] = extractelement <4 x i32> [[TMP1]], i64 1
; CHECK-NEXT:    ret void
;
entry:
  %interleaved.vec = load <8 x i32>, <8 x i32>* %ptr, align 8
  %v0 = shufflevector <8 x i32> %interleaved.vec, <8 x i32> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  br i1 %c, label %if.then, label %if.merge

if.then:
  %e0 = extractelement <8 x i32> %interleaved.vec, i32 0
  br label %if.merge

if.merge:
  %e1 = extractelement <8 x i32> %interleaved.vec, i32 2
  ret void
}

define void @extract_user_multi_no_dom(<8 x i32>* %ptr, i1 %c) {
; CHECK-LABEL: @extract_user_multi_no_dom(
; CHECK-NOT:     @llvm.aarch64.neon
; CHECK:         ret void
;
entry:
  %interleaved.vec = load <8 x i32>, <8 x i32>* %ptr, align 8
  %e0 = extractelement <8 x i32> %interleaved.vec, i32 0
  br i1 %c, label %if.then, label %if.merge

if.then:
  %v0 = shufflevector <8 x i32> %interleaved.vec, <8 x i32> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %e1 = extractelement <8 x i32> %interleaved.vec, i32 2
  br label %if.merge

if.merge:
  ret void
}

define void @extract_user_wrong_const_index(<8 x i32>* %ptr) {
; CHECK-LABEL: @extract_user_wrong_const_index(
; CHECK-NOT:     @llvm.aarch64.neon
; CHECK:         ret void
;
entry:
  %interleaved.vec = load <8 x i32>, <8 x i32>* %ptr, align 8
  %v0 = shufflevector <8 x i32> %interleaved.vec, <8 x i32> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %e0 = extractelement <8 x i32> %interleaved.vec, i32 1
  ret void
}

define void @extract_user_undef_index(<8 x i32>* %ptr) {
; CHECK-LABEL: @extract_user_undef_index(
; CHECK-NOT:     @llvm.aarch64.neon
; CHECK:         ret void
;
entry:
  %interleaved.vec = load <8 x i32>, <8 x i32>* %ptr, align 8
  %v0 = shufflevector <8 x i32> %interleaved.vec, <8 x i32> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %e0 = extractelement <8 x i32> %interleaved.vec, i32 undef
  ret void
}

define void @extract_user_var_index(<8 x i32>* %ptr, i32 %i) {
; CHECK-LABEL: @extract_user_var_index(
; CHECK-NOT:     @llvm.aarch64.neon
; CHECK:         ret void
;
entry:
  %interleaved.vec = load <8 x i32>, <8 x i32>* %ptr, align 8
  %v0 = shufflevector <8 x i32> %interleaved.vec, <8 x i32> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %e0 = extractelement <8 x i32> %interleaved.vec, i32 %i
  ret void
}
