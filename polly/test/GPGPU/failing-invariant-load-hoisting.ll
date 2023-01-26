; RUN: opt %loadPolly -S < %s -polly-codegen-ppcg \
; RUN: -polly-invariant-load-hoisting | FileCheck %s -check-prefix=CODEGEN

; REQUIRES: pollyacc

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n8:16:32-S64"

%S = type { i32, i32, [12 x %L] }
%L = type { i32, i32, double, i32, i32, i32, i32, i32 }

define void @test(ptr %cpi, i1 %b) {
; CODEGEN-LABEL: @test(
; CODEGEN:    polly.preload.begin:
; CODEGEN-NEXT:  br i1 false

entry:
  %nt = getelementptr inbounds %S, ptr %cpi, i32 0, i32 1
  br i1 %b, label %if.then14, label %exit

if.then14:
  %l0 = load i32, ptr %cpi, align 8
  %cmp12.i = icmp sgt i32 %l0, 0
  br i1 %cmp12.i, label %for.body.lr.ph.i, label %exit

for.body.lr.ph.i:
  %l1 = load i32, ptr %nt, align 4
  br label %for.body.i

for.body.i:
  %phi = phi i32 [ 0, %for.body.lr.ph.i ], [ %inc, %for.body.i ]
  %mul.i163 = mul nsw i32 %phi, %l1
  %cv = getelementptr inbounds %S, ptr %cpi, i32 0, i32 2, i32 %mul.i163, i32 0
  store i32 0, ptr %cv, align 8
  %inc = add nuw nsw i32 %phi, 1
  %l2 = load i32, ptr %cpi, align 8
  %cmp.i164 = icmp slt i32 %inc, %l2
  br i1 %cmp.i164, label %for.body.i, label %exit

exit:
  ret void
}
