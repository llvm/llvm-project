; RUN: opt -S < %s -passes=loop-vectorize -force-vector-width=4 | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

; CHECK-LABEL: test0
define void @test0(i16* noalias %M3) {
entry:
  br label %if.then1165.us

if.then1165.us:                                   ; preds = %if.then1165.us, %entry
  %indvars.iv1783 = phi i64 [ 0, %entry ], [ %indvars.iv.next1784, %if.then1165.us ]
  %conv1177.us = zext i16 undef to i32
  %add1178.us = add nsw i32 %conv1177.us, undef
  %conv1179.us = trunc i32 %add1178.us to i16
  %idxprom1181.us = ashr exact i64 undef, 32
  %arrayidx1185.us = getelementptr inbounds i16, i16* %M3, i64 %idxprom1181.us
  store i16 %conv1179.us, i16* %arrayidx1185.us, align 2
  %indvars.iv.next1784 = add nuw nsw i64 %indvars.iv1783, 1
  %exitcond1785 = icmp eq i64 %indvars.iv.next1784, 16
  br i1 %exitcond1785, label %for.inc1286.loopexit, label %if.then1165.us

for.inc1286.loopexit:                             ; preds = %if.then1165.us
  ret void
}

; CHECK-LABEL: test1
define void @test1(i16* noalias %M3) {
entry:
  br label %if.then1165.us

if.then1165.us:                                   ; preds = %if.then1165.us, %entry
  %indvars.iv1783 = phi i64 [ 0, %entry ], [ %indvars.iv.next1784, %if.then1165.us ]
  %fptr = load i32, i32* undef, align 4
  %conv1177.us = zext i16 undef to i32
  %add1178.us = add nsw i32 %conv1177.us, %fptr
  %conv1179.us = trunc i32 %add1178.us to i16
  %idxprom1181.us = ashr exact i64 undef, 32
  %arrayidx1185.us = getelementptr inbounds i16, i16* %M3, i64 %idxprom1181.us
  store i16 %conv1179.us, i16* %arrayidx1185.us, align 2
  %indvars.iv.next1784 = add nuw nsw i64 %indvars.iv1783, 1
  %exitcond1785 = icmp eq i64 %indvars.iv.next1784, 16
  br i1 %exitcond1785, label %for.inc1286.loopexit, label %if.then1165.us

for.inc1286.loopexit:                             ; preds = %if.then1165.us
  ret void
}
