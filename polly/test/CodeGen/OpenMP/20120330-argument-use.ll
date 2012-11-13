; RUN: opt %loadPolly %defaultOpts -polly-codegen -enable-polly-openmp < %s -S | FileCheck %s

;/*
; * =============================================================================
; *
; *       Filename:  20120330-argument-use.c
; *
; *    Description:  Polly OpenMP test case
; *
; *                  Test if the OpenMP subfunction uses the argument copy in
; *                  the OpenMP struct not the original one only available in
; *                  the original function.
; *
; *                  Run with -polly-codegen -enable-polly-openmp
; *
; *         Author:  Johannes Doerfert johannes@jdoerfert.de
; *
; *        Created:  2012-03-30
; *       Modified:  2012-03-30
; *
; * =============================================================================
; */
;
;void f(int * restrict A, int * restrict B, int n) {
;  int i;
;
;  for (i = 0; i < n; i++) {
;    A[i] = B[i] * 2;
;  }
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @f(i32* noalias %A, i32* noalias %B, i32 %n) nounwind uwtable {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %tmp = trunc i64 %indvars.iv to i32
  %cmp = icmp slt i32 %tmp, %n
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32* %B, i64 %indvars.iv
  %tmp1 = load i32* %arrayidx, align 4
  %mul = shl nsw i32 %tmp1, 1
  %arrayidx2 = getelementptr inbounds i32* %A, i64 %indvars.iv
  store i32 %mul, i32* %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; CHECK: %omp.userContext1 = bitcast i8* %omp.userContext to { i32, i32, i32*, i32* }*
; CHECK: %0 = getelementptr inbounds { i32, i32, i32*, i32* }* %omp.userContext1, i32 0, i32 0
; CHECK: %1 = load i32* %0
; CHECK: %2 = getelementptr inbounds { i32, i32, i32*, i32* }* %omp.userContext1, i32 0, i32 1
; CHECK: %3 = load i32* %2
; CHECK: %4 = getelementptr inbounds { i32, i32, i32*, i32* }* %omp.userContext1, i32 0, i32 2
; CHECK: %5 = load i32** %4
; CHECK: %6 = getelementptr inbounds { i32, i32, i32*, i32* }* %omp.userContext1, i32 0, i32 3
; CHECK: %7 = load i32** %6

