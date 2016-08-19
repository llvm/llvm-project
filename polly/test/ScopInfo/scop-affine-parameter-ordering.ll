; RUN: opt -polly-process-unprofitable -polly-scops %s -analyze | FileCheck %s
target datalayout = "e-m:e-i64:64-i128:128-n8:16:32:64-S128"
target triple = "aarch64--linux-android"

; Check for SCOP:
; CHECK: Stmt_for_body8_us_us95_i
; CHECK-NEXT: Domain :=
; CHECK-NEXT: [p_0] -> { Stmt_for_body8_us_us95_i[i0] : 0 <= i0 <= 4 };
; CHECK-NEXT: Schedule :=
; CHECK-NEXT: [p_0] -> { Stmt_for_body8_us_us95_i[i0] -> [i0] };
; CHECK-NEXT; MustWriteAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT; [p_0] -> { Stmt_for_body8_us_us95_i[i0] -> MemRef_0[1 + p_0] };
; CHECK-NEXT }

define void @test1() unnamed_addr align 2 {
entry:
  %xFactor.0.reload = load i32, i32* undef
  %0 = load i8*, i8** undef, align 8
  %div = udiv i32 0, %xFactor.0.reload
  %1 = load i32, i32* undef, align 4
  %mul = mul i32 %1, %xFactor.0.reload
  %col.023.us.us85.i = add i32 %div, -1
  %mul11.us.us93.i = mul i32 %col.023.us.us85.i, %mul
  br label %for.body8.us.us95.i

for.body8.us.us95.i:
  %niter.i = phi i32 [ %niter.i.next, %for.body8.us.us95.i ], [ 0, %entry ]
  %add12.us.us100.1.i = add i32 1, %mul11.us.us93.i
  %idxprom13.us.us101.1.i = zext i32 %add12.us.us100.1.i to i64
  %arrayidx14.us.us102.1.i = getelementptr inbounds i8, i8* %0, i64 %idxprom13.us.us101.1.i
  store i8 0, i8* %arrayidx14.us.us102.1.i, align 1
  %niter.ncmp.3.i = icmp eq i32 %niter.i, 4
  %niter.i.next = add i32 %niter.i, 1
  br i1 %niter.ncmp.3.i, label %for.body8.us.us95.epil.i.preheader, label %for.body8.us.us95.i

for.body8.us.us95.epil.i.preheader:
  ret void
}

