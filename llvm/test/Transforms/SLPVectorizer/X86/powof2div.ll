; RUN: opt < %s -basicaa -slp-vectorizer -S -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

;CHECK-LABEL: @powof2div(
;CHECK: load <4 x i32>, <4 x i32>*
;CHECK: add nsw <4 x i32>
;CHECK: sdiv <4 x i32>
define void @powof2div(i32* noalias nocapture %a, i32* noalias nocapture readonly %b, i32* noalias nocapture readonly %c){
entry:
  %0 = load i32, i32* %b, align 4
  %1 = load i32, i32* %c, align 4
  %add = add nsw i32 %1, %0
  %div = sdiv i32 %add, 2
  store i32 %div, i32* %a, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %b, i64 1
  %2 = load i32, i32* %arrayidx3, align 4
  %arrayidx4 = getelementptr inbounds i32, i32* %c, i64 1
  %3 = load i32, i32* %arrayidx4, align 4
  %add5 = add nsw i32 %3, %2
  %div6 = sdiv i32 %add5, 2
  %arrayidx7 = getelementptr inbounds i32, i32* %a, i64 1
  store i32 %div6, i32* %arrayidx7, align 4
  %arrayidx8 = getelementptr inbounds i32, i32* %b, i64 2
  %4 = load i32, i32* %arrayidx8, align 4
  %arrayidx9 = getelementptr inbounds i32, i32* %c, i64 2
  %5 = load i32, i32* %arrayidx9, align 4
  %add10 = add nsw i32 %5, %4
  %div11 = sdiv i32 %add10, 2
  %arrayidx12 = getelementptr inbounds i32, i32* %a, i64 2
  store i32 %div11, i32* %arrayidx12, align 4
  %arrayidx13 = getelementptr inbounds i32, i32* %b, i64 3
  %6 = load i32, i32* %arrayidx13, align 4
  %arrayidx14 = getelementptr inbounds i32, i32* %c, i64 3
  %7 = load i32, i32* %arrayidx14, align 4
  %add15 = add nsw i32 %7, %6
  %div16 = sdiv i32 %add15, 2
  %arrayidx17 = getelementptr inbounds i32, i32* %a, i64 3
  store i32 %div16, i32* %arrayidx17, align 4
  ret void
}

