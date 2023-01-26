; RUN: opt %loadPolly -polly-process-unprofitable -polly-invariant-load-hoisting -polly-print-scops -disable-output < %s | FileCheck %s -check-prefix=SCOPS
; RUN: opt %loadPolly -S < %s -polly-codegen -polly-process-unprofitable -polly-invariant-load-hoisting | FileCheck %s -check-prefix=CODEGEN

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n8:16:32-S64"

%S = type { i32, i32, [12 x %L] }
%L = type { i32, i32, double, i32, i32, i32, i32, i32 }

define void @test(ptr %cpi, i1 %b) {
; SCOPS-LABEL: Region: %if.then14---%exit
; SCOPS:         Invariant Accesses: {
; SCOPS-NEXT:            ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; SCOPS-NEXT:                [l2, l1] -> { Stmt_for_body_i[i0] -> MemRef_cpi[0, 0] };
; SCOPS-NEXT:            Execution Context: [l2, l1] -> {  :  }
; SCOPS-NEXT:            ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; SCOPS-NEXT:                [l2, l1] -> { Stmt_for_body_lr_ph_i[] -> MemRef_cpi[0, 1] };
; SCOPS-NEXT:            Execution Context: [l2, l1] -> {  : l2 > 0 }
; SCOPS-NEXT:    }
; SCOPS:         Arrays {
; SCOPS-NEXT:        i32 MemRef_cpi[*][(10 * %l1)]; // Element size 4
; SCOPS-NEXT:    }

; FIXME: Figure out how to actually generate code for this loop.
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
