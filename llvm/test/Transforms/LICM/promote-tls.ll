; RUN: opt -tbaa -basicaa -licm -S < %s | FileCheck %s
; RUN: opt -aa-pipeline=type-based-aa,basic-aa -passes='require<aa>,require<targetir>,require<scalar-evolution>,require<opt-remark-emit>,loop(licm)' -S %s | FileCheck %s

; If we can prove a local is thread local, we can insert stores during
; promotion which wouldn't be legal otherwise.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-linux-generic"

@p = external global i8*

declare i8* @malloc(i64)

; Exercise the TLS case
; CHECK-LABEL: @test
define i32* @test(i32 %n) {
entry:
  ;; ignore the required null check for simplicity
  %mem = call dereferenceable(16) noalias i8* @malloc(i64 16)
  %addr = bitcast i8* %mem to i32*
  br label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  br label %for.header

for.header:
  %i.02 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %old = load i32, i32* %addr, align 4
  ; deliberate impossible to analyze branch
  %guard = load atomic i8*, i8** @p monotonic, align 8
  %exitcmp = icmp eq i8* %guard, null
  br i1 %exitcmp, label %for.body, label %early-exit

early-exit:
; CHECK-LABEL: early-exit:
; CHECK: store i32 %new1.lcssa, i32* %addr, align 1
  ret i32* null

for.body:
  %new = add i32 %old, 1
  store i32 %new, i32* %addr, align 4
  %inc = add nsw i32 %i.02, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %for.header, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.body
; CHECK-LABEL: for.cond.for.end_crit_edge:
; CHECK: store i32 %new.lcssa, i32* %addr, align 1
  %split = phi i32* [ %addr, %for.body ]
  ret i32* null
}

; Stack allocations can also be thread-local
; CHECK-LABEL: @test2
define i32* @test2(i32 %n) {
entry:
  %mem = alloca i8, i32 16
  %addr = bitcast i8* %mem to i32*
  br label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  br label %for.header

for.header:
  %i.02 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %old = load i32, i32* %addr, align 4
  ; deliberate impossible to analyze branch
  %guard = load atomic i8*, i8** @p monotonic, align 8
  %exitcmp = icmp eq i8* %guard, null
  br i1 %exitcmp, label %for.body, label %early-exit

early-exit:
; CHECK-LABEL: early-exit:
; CHECK: store i32 %new1.lcssa, i32* %addr, align 1
  ret i32* null

for.body:
  %new = add i32 %old, 1
  store i32 %new, i32* %addr, align 4
  %inc = add nsw i32 %i.02, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %for.header, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.body
; CHECK-LABEL: for.cond.for.end_crit_edge:
; CHECK: store i32 %new.lcssa, i32* %addr, align 1
  %split = phi i32* [ %addr, %for.body ]
  ret i32* null
}

declare i8* @not_malloc(i64)

; Negative test - not TLS
; CHECK-LABEL: @test_neg
define i32* @test_neg(i32 %n) {
entry:
  ;; ignore the required null check for simplicity
  %mem = call dereferenceable(16) noalias i8* @not_malloc(i64 16)
  %addr = bitcast i8* %mem to i32*
  br label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  br label %for.header

for.header:
  %i.02 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %old = load i32, i32* %addr, align 4
  ; deliberate impossible to analyze branch
  %guard = load volatile i8*, i8** @p
  %exitcmp = icmp eq i8* %guard, null
  br i1 %exitcmp, label %for.body, label %early-exit

early-exit:
; CHECK-LABEL: early-exit:
; CHECK-NOT: store
  ret i32* null

for.body:
; CHECK-LABEL: for.body:
; CHECK: store i32 %new, i32* %addr, align 4
  %new = add i32 %old, 1
  store i32 %new, i32* %addr, align 4
  %inc = add nsw i32 %i.02, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %for.header, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.body
; CHECK-LABEL: for.cond.for.end_crit_edge:
; CHECK-NOT: store
  %split = phi i32* [ %addr, %for.body ]
  ret i32* null
}

; Negative test - can't speculate load since branch
; may control alignment
; CHECK-LABEL: @test_neg2
define i32* @test_neg2(i32 %n) {
entry:
  ;; ignore the required null check for simplicity
  %mem = call dereferenceable(16) noalias i8* @malloc(i64 16)
  %addr = bitcast i8* %mem to i32*
  br label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  br label %for.header

for.header:
  %i.02 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  ; deliberate impossible to analyze branch
  %guard = load volatile i8*, i8** @p
  %exitcmp = icmp eq i8* %guard, null
  br i1 %exitcmp, label %for.body, label %early-exit

early-exit:
; CHECK-LABEL: early-exit:
; CHECK-NOT: store
  ret i32* null

for.body:
; CHECK-LABEL: for.body:
; CHECK: store i32 %new, i32* %addr, align 4
  %old = load i32, i32* %addr, align 4
  %new = add i32 %old, 1
  store i32 %new, i32* %addr, align 4
  %inc = add nsw i32 %i.02, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %for.header, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.body
; CHECK-LABEL: for.cond.for.end_crit_edge:
; CHECK-NOT: store
  %split = phi i32* [ %addr, %for.body ]
  ret i32* null
}
