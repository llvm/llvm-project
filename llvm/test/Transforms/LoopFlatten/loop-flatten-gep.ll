; RUN: opt < %s -S -passes='loop(loop-flatten),verify' -verify-loop-info -verify-dom-info -verify-scev | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

; We should be able to flatten the loops and turn the two geps into one.
; CHECK-LABEL: test1
define void @test1(i32 %N, ptr %A) {
entry:
  %cmp3 = icmp ult i32 0, %N
  br i1 %cmp3, label %for.outer.preheader, label %for.end

; CHECK-LABEL: for.outer.preheader:
; CHECK: %flatten.tripcount = mul i32 %N, %N
for.outer.preheader:
  br label %for.inner.preheader

; CHECK-LABEL: for.inner.preheader:
; CHECK: %flatten.arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
for.inner.preheader:
  %i = phi i32 [ 0, %for.outer.preheader ], [ %inc2, %for.outer ]
  br label %for.inner

; CHECK-LABEL: for.inner:
; CHECK: store i32 0, ptr %flatten.arrayidx, align 4
; CHECK: br label %for.outer
for.inner:
  %j = phi i32 [ 0, %for.inner.preheader ], [ %inc1, %for.inner ]
  %mul = mul i32 %i, %N
  %gep = getelementptr inbounds i32, ptr %A, i32 %mul
  %arrayidx = getelementptr inbounds i32, ptr %gep, i32 %j
  store i32 0, ptr %arrayidx, align 4
  %inc1 = add nuw i32 %j, 1
  %cmp2 = icmp ult i32 %inc1, %N
  br i1 %cmp2, label %for.inner, label %for.outer

; CHECK-LABEL: for.outer:
; CHECK: %cmp1 = icmp ult i32 %inc2, %flatten.tripcount
for.outer:
  %inc2 = add i32 %i, 1
  %cmp1 = icmp ult i32 %inc2, %N
  br i1 %cmp1, label %for.inner.preheader, label %for.end.loopexit

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

; We can flatten, but the flattened gep has to be inserted after the load it
; depends on.
; CHECK-LABEL: test2
define void @test2(i32 %N, ptr %A) {
entry:
  %cmp3 = icmp ult i32 0, %N
  br i1 %cmp3, label %for.outer.preheader, label %for.end

; CHECK-LABEL: for.outer.preheader:
; CHECK: %flatten.tripcount = mul i32 %N, %N
for.outer.preheader:
  br label %for.inner.preheader

; CHECK-LABEL: for.inner.preheader:
; CHECK-NOT: getelementptr inbounds i32, ptr %ptr, i32 %i
for.inner.preheader:
  %i = phi i32 [ 0, %for.outer.preheader ], [ %inc2, %for.outer ]
  br label %for.inner

; CHECK-LABEL: for.inner:
; CHECK: %flatten.arrayidx = getelementptr inbounds i32, ptr %ptr, i32 %i
; CHECK: store i32 0, ptr %flatten.arrayidx, align 4
; CHECK: br label %for.outer
for.inner:
  %j = phi i32 [ 0, %for.inner.preheader ], [ %inc1, %for.inner ]
  %ptr = load volatile ptr, ptr %A, align 4
  %mul = mul i32 %i, %N
  %gep = getelementptr inbounds i32, ptr %ptr, i32 %mul
  %arrayidx = getelementptr inbounds i32, ptr %gep, i32 %j
  store i32 0, ptr %arrayidx, align 4
  %inc1 = add nuw i32 %j, 1
  %cmp2 = icmp ult i32 %inc1, %N
  br i1 %cmp2, label %for.inner, label %for.outer

; CHECK-LABEL: for.outer:
; CHECK: %cmp1 = icmp ult i32 %inc2, %flatten.tripcount
for.outer:
  %inc2 = add i32 %i, 1
  %cmp1 = icmp ult i32 %inc2, %N
  br i1 %cmp1, label %for.inner.preheader, label %for.end.loopexit

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

; We can't flatten if the gep offset is smaller than the pointer size.
; CHECK-LABEL: test3
define void @test3(i16 %N, ptr %A) {
entry:
  %cmp3 = icmp ult i16 0, %N
  br i1 %cmp3, label %for.outer.preheader, label %for.end

for.outer.preheader:
  br label %for.inner.preheader

; CHECK-LABEL: for.inner.preheader:
; CHECK-NOT: getelementptr i32, ptr %A, i16 %i
for.inner.preheader:
  %i = phi i16 [ 0, %for.outer.preheader ], [ %inc2, %for.outer ]
  br label %for.inner

; CHECK-LABEL: for.inner:
; CHECK-NOT: getelementptr i32, ptr %A, i16 %i
; CHECK: br i1 %cmp2, label %for.inner, label %for.outer
for.inner:
  %j = phi i16 [ 0, %for.inner.preheader ], [ %inc1, %for.inner ]
  %mul = mul i16 %i, %N
  %gep = getelementptr inbounds i32, ptr %A, i16 %mul
  %arrayidx = getelementptr inbounds i32, ptr %gep, i16 %j
  store i32 0, ptr %arrayidx, align 4
  %inc1 = add nuw i16 %j, 1
  %cmp2 = icmp ult i16 %inc1, %N
  br i1 %cmp2, label %for.inner, label %for.outer

for.outer:
  %inc2 = add i16 %i, 1
  %cmp1 = icmp ult i16 %inc2, %N
  br i1 %cmp1, label %for.inner.preheader, label %for.end.loopexit

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}
