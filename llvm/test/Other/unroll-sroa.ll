; RUN: opt -disable-verify -passes='default<O2>' -S < %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; The local array %tmp can only be optimized away by sroa after loop unroll.

; CHECK-LABEL: define void @foo
; CHECK-NOT:   alloca
; CHECK-NOT:   call void @llvm.memcpy.p0i8.p0i8.i64

; Function Attrs: nounwind uwtable
define void @foo(i32* %a, i32* %b) {
entry:
  %a.addr = alloca i32*, align 8
  %b.addr = alloca i32*, align 8
  %tmp = alloca [4 x float], align 16
  %i = alloca i32, align 4
  store i32* %a, i32** %a.addr, align 8
  store i32* %b, i32** %b.addr, align 8
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %iter2 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %iter2, 4
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  br label %for.end

for.body:                                         ; preds = %for.cond
  %inptr = load i32*, i32** %a.addr, align 8
  %idx2 = load i32, i32* %i, align 4
  %idxprom = sext i32 %idx2 to i64
  %arrayidx = getelementptr inbounds i32, i32* %inptr, i64 %idxprom
  %val = load i32, i32* %arrayidx, align 4
  %conv = sitofp i32 %val to float
  %idx = load i32, i32* %i, align 4
  %idxprom1 = sext i32 %idx to i64
  %arrayidx2 = getelementptr inbounds [4 x float], [4 x float]* %tmp, i64 0, i64 %idxprom1
  store float %conv, float* %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %iter = load i32, i32* %i, align 4
  %inc = add nsw i32 %iter, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond.cleanup
  %dstptr = load i32*, i32** %b.addr, align 8
  %dst = bitcast i32* %dstptr to i8*
  %arraydecay = getelementptr inbounds [4 x float], [4 x float]* %tmp, i64 0, i64 0
  %src = bitcast float* %arraydecay to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %dst, i8* align 16 %src, i64 16, i1 false)
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg)
