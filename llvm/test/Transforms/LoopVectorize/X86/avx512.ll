; RUN: opt -mattr=+avx512f -passes=loop-vectorize -S < %s | FileCheck %s --check-prefixes=CHECK,CHECK-NO-PREFER
; RUN: opt -mattr=+avx512vl,+prefer-256-bit -passes=loop-vectorize -S < %s | FileCheck %s --check-prefixes=CHECK,CHECK-PREFER-AVX256

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

; Verify that we generate 512-bit wide vectors for a basic integer memset
; loop.

; CHECK-NO-PREFER-LABEL: @f(
; CHECK-NO-PREFER: vector.body:
; CHECK-NO-PREFER: store <16 x i32>
; CHECK-NO-PREFER: vec.epilog.vector.body:
; CHECK-NO-PREFER: store <8 x i32>

; Verify that we don't generate 512-bit wide vectors when subtarget feature says not to

; CHECK-PREFER-AVX256-LABEL: @f(
; CHECK-PREFER-AVX256: vector.body:
; CHECK-PREFER-AVX256: store <8 x i32>
; CHECK-PREFER-AVX256: vec.epilog.vector.body:
; CHECK-PREFER-AVX256: store <4 x i32>

define void @f(ptr %a, i32 %n) {
entry:
  %cmp4 = icmp sgt i32 %n, 0
  br i1 %cmp4, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  store i32 %n, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}

; Verify that the "prefer-vector-width=256" attribute prevents the use of 512-bit
; vectors

; CHECK-LABEL: @g(
; CHECK: vector.body:
; CHECK: store <8 x i32>
; CHECK: vec.epilog.vector.body:
; CHECK: store <4 x i32>

define void @g(ptr %a, i32 %n) "prefer-vector-width"="256" {
entry:
  %cmp4 = icmp sgt i32 %n, 0
  br i1 %cmp4, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  store i32 %n, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}

; Verify that the "prefer-vector-width=512" attribute override the subtarget
; vectors

; CHECK-LABEL: @h(
; CHECK: vector.body:
; CHECK: store <16 x i32>
; CHECK: vec.epilog.vector.body:
; CHECK: store <8 x i32>

define void @h(ptr %a, i32 %n) "prefer-vector-width"="512" {
entry:
  %cmp4 = icmp sgt i32 %n, 0
  br i1 %cmp4, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  store i32 %n, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}
