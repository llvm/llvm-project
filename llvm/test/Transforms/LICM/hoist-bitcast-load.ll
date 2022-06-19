; RUN: opt -aa-pipeline=basic-aa -passes='require<opt-remark-emit>,loop-mssa(loop-simplifycfg,licm)' -S < %s | FileCheck %s
; RUN: opt -S -basic-aa -licm -verify-memoryssa < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Make sure the basic alloca pointer hoisting works:
; CHECK-LABEL: @test1
; CHECK: load i32, i32* %c, align 4
; CHECK: for.body:

; Function Attrs: nounwind uwtable
define void @test1(i32* nocapture %a, i32* nocapture readonly %b, i32 %n) #0 {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  %c = alloca i32
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.inc
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 0
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %1 = load i32, i32* %c, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %2 = load i32, i32* %arrayidx3, align 4
  %mul = mul nsw i32 %2, %1
  store i32 %mul, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry
  ret void
}

; Make sure the basic alloca pointer hoisting works through a bitcast to a
; pointer to a smaller type:
; CHECK-LABEL: @test2
; CHECK: load i32, i32* %c, align 4
; CHECK: for.body:

; Function Attrs: nounwind uwtable
define void @test2(i32* nocapture %a, i32* nocapture readonly %b, i32 %n) #0 {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  %ca = alloca i64
  %c = bitcast i64* %ca to i32*
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.inc
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 0
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %1 = load i32, i32* %c, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %2 = load i32, i32* %arrayidx3, align 4
  %mul = mul nsw i32 %2, %1
  store i32 %mul, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry
  ret void
}

; Make sure the basic alloca pointer hoisting works through an addrspacecast
; CHECK-LABEL: @test2_addrspacecast
; CHECK: load i32, i32 addrspace(1)* %c, align 4
; CHECK: for.body:

; Function Attrs: nounwind uwtable
define void @test2_addrspacecast(i32 addrspace(1)* nocapture %a, i32 addrspace(1)* nocapture readonly %b, i32 %n) #0 {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  %ca = alloca i64
  %c = addrspacecast i64* %ca to i32 addrspace(1)*
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.inc
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %a, i64 %indvars.iv
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 0
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %1 = load i32, i32 addrspace(1)* %c, align 4
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %b, i64 %indvars.iv
  %2 = load i32, i32 addrspace(1)* %arrayidx3, align 4
  %mul = mul nsw i32 %2, %1
  store i32 %mul, i32 addrspace(1)* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry
  ret void
}

; Make sure the basic alloca pointer hoisting works through a bitcast to a
; pointer to a smaller type (where the bitcast also needs to be hoisted):
; CHECK-LABEL: @test3
; CHECK: load i32, i32* %c, align 4
; CHECK: for.body:

; Function Attrs: nounwind uwtable
define void @test3(i32* nocapture %a, i32* nocapture readonly %b, i32 %n) #0 {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  %ca = alloca i64
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.inc
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 0
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %c = bitcast i64* %ca to i32*
  %1 = load i32, i32* %c, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %2 = load i32, i32* %arrayidx3, align 4
  %mul = mul nsw i32 %2, %1
  store i32 %mul, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry
  ret void
}

; Make sure the basic alloca pointer hoisting does not happen through a bitcast
; to a pointer to a larger type:
; CHECK-LABEL: @test4
; CHECK: for.body:
; CHECK: load i32, i32* %c, align 4

; Function Attrs: nounwind uwtable
define void @test4(i32* nocapture %a, i32* nocapture readonly %b, i32 %n) #0 {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  %ca = alloca i16
  %c = bitcast i16* %ca to i32*
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.inc
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 0
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %1 = load i32, i32* %c, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %2 = load i32, i32* %arrayidx3, align 4
  %mul = mul nsw i32 %2, %1
  store i32 %mul, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry
  ret void
}

; Don't crash on bitcasts to unsized types.
; CHECK-LABEL: @test5
; CHECK: for.body:
; CHECK: load i32, i32* %c, align 4

%atype = type opaque

; Function Attrs: nounwind uwtable
define void @test5(i32* nocapture %a, i32* nocapture readonly %b, i32 %n) #0 {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  %ca = alloca i16
  %cab = bitcast i16* %ca to %atype*
  %c = bitcast %atype* %cab to i32*
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.inc
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 0
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %1 = load i32, i32* %c, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %2 = load i32, i32* %arrayidx3, align 4
  %mul = mul nsw i32 %2, %1
  store i32 %mul, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry
  ret void
}

attributes #0 = { nounwind uwtable }

