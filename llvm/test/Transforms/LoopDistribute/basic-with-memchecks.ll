; RUN: opt -basicaa -loop-distribute -enable-loop-distribute -verify-loop-info -verify-dom-info -S \
; RUN:   < %s | FileCheck %s

; RUN: opt -basicaa -loop-distribute -enable-loop-distribute -loop-vectorize -force-vector-width=4 \
; RUN:   -verify-loop-info -verify-dom-info -S < %s | \
; RUN:   FileCheck --check-prefix=VECTORIZE %s

; RUN: opt -basicaa -loop-distribute -enable-loop-distribute -verify-loop-info -verify-dom-info \
; RUN:   -loop-accesses -analyze < %s | FileCheck %s --check-prefix=ANALYSIS

; The memcheck version of basic.ll.  We should distribute and vectorize the
; second part of this loop with 5 memchecks (A+1 x {C, D, E} + C x {A, B})
;
;   for (i = 0; i < n; i++) {
;     A[i + 1] = A[i] * B[i];
; -------------------------------
;     C[i] = D[i] * E[i];
;   }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

@B = common global i32* null, align 8
@A = common global i32* null, align 8
@C = common global i32* null, align 8
@D = common global i32* null, align 8
@E = common global i32* null, align 8

; CHECK-LABEL: @f(
define void @f() {
entry:
  %a = load i32*, i32** @A, align 8
  %b = load i32*, i32** @B, align 8
  %c = load i32*, i32** @C, align 8
  %d = load i32*, i32** @D, align 8
  %e = load i32*, i32** @E, align 8
  br label %for.body

; We have two compares for each array overlap check.
; Since the checks to A and A + 4 get merged, this will give us a
; total of 8 compares.
;
; CHECK: for.body.lver.check:
; CHECK:     = icmp
; CHECK:     = icmp

; CHECK:     = icmp
; CHECK:     = icmp

; CHECK:     = icmp
; CHECK:     = icmp

; CHECK:     = icmp
; CHECK:     = icmp

; CHECK-NOT: = icmp
; CHECK:     br i1 %memcheck.conflict, label %for.body.ph.lver.orig, label %for.body.ph.ldist1

; The non-distributed loop that the memchecks fall back on.

; CHECK: for.body.ph.lver.orig:
; CHECK:     br label %for.body.lver.orig
; CHECK: for.body.lver.orig:
; CHECK:    br i1 %exitcond.lver.orig, label %for.end, label %for.body.lver.orig

; Verify the two distributed loops.

; CHECK: for.body.ph.ldist1:
; CHECK:     br label %for.body.ldist1
; CHECK: for.body.ldist1:
; CHECK:    %mulA.ldist1 = mul i32 %loadB.ldist1, %loadA.ldist1
; CHECK:    br i1 %exitcond.ldist1, label %for.body.ph, label %for.body.ldist1

; CHECK: for.body.ph:
; CHECK:    br label %for.body
; CHECK: for.body:
; CHECK:    %mulC = mul i32 %loadD, %loadE
; CHECK: for.end:


; VECTORIZE: mul <4 x i32>

for.body:                                         ; preds = %for.body, %entry
  %ind = phi i64 [ 0, %entry ], [ %add, %for.body ]

  %arrayidxA = getelementptr inbounds i32, i32* %a, i64 %ind
  %loadA = load i32, i32* %arrayidxA, align 4

  %arrayidxB = getelementptr inbounds i32, i32* %b, i64 %ind
  %loadB = load i32, i32* %arrayidxB, align 4

  %mulA = mul i32 %loadB, %loadA

  %add = add nuw nsw i64 %ind, 1
  %arrayidxA_plus_4 = getelementptr inbounds i32, i32* %a, i64 %add
  store i32 %mulA, i32* %arrayidxA_plus_4, align 4

  %arrayidxD = getelementptr inbounds i32, i32* %d, i64 %ind
  %loadD = load i32, i32* %arrayidxD, align 4

  %arrayidxE = getelementptr inbounds i32, i32* %e, i64 %ind
  %loadE = load i32, i32* %arrayidxE, align 4

  %mulC = mul i32 %loadD, %loadE

  %arrayidxC = getelementptr inbounds i32, i32* %c, i64 %ind
  store i32 %mulC, i32* %arrayidxC, align 4

  %exitcond = icmp eq i64 %add, 20
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; Make sure there's no "Multiple reports generated" assert with a
; volatile load, and no distribution

; TODO: Distribution of volatile may be possible under some
; circumstance, but the current implementation does not touch them.

; CHECK-LABEL: @f_volatile_load(
; CHECK: br label %for.body{{$}}

; CHECK-NOT: load

; CHECK: {{^}}for.body:
; CHECK: load i32
; CHECK: load i32
; CHECK: load volatile i32
; CHECK: load i32
; CHECK: br i1 %exitcond, label %for.end, label %for.body{{$}}

; CHECK-NOT: load

; VECTORIZE-NOT: load <4 x i32>
; VECTORIZE-NOT: mul <4 x i32>
define void @f_volatile_load() {
entry:
  %a = load i32*, i32** @A, align 8
  %b = load i32*, i32** @B, align 8
  %c = load i32*, i32** @C, align 8
  %d = load i32*, i32** @D, align 8
  %e = load i32*, i32** @E, align 8
  br label %for.body

for.body:
  %ind = phi i64 [ 0, %entry ], [ %add, %for.body ]

  %arrayidxA = getelementptr inbounds i32, i32* %a, i64 %ind
  %loadA = load i32, i32* %arrayidxA, align 4

  %arrayidxB = getelementptr inbounds i32, i32* %b, i64 %ind
  %loadB = load i32, i32* %arrayidxB, align 4

  %mulA = mul i32 %loadB, %loadA

  %add = add nuw nsw i64 %ind, 1
  %arrayidxA_plus_4 = getelementptr inbounds i32, i32* %a, i64 %add
  store i32 %mulA, i32* %arrayidxA_plus_4, align 4

  %arrayidxD = getelementptr inbounds i32, i32* %d, i64 %ind
  %loadD = load volatile i32, i32* %arrayidxD, align 4

  %arrayidxE = getelementptr inbounds i32, i32* %e, i64 %ind
  %loadE = load i32, i32* %arrayidxE, align 4

  %mulC = mul i32 %loadD, %loadE

  %arrayidxC = getelementptr inbounds i32, i32* %c, i64 %ind
  store i32 %mulC, i32* %arrayidxC, align 4

  %exitcond = icmp eq i64 %add, 20
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare i32 @llvm.convergent(i32) #0

; This is the same as f, and would require the same bounds
; check. However, it is not OK to introduce new control dependencies
; on the convergent call.

; CHECK-LABEL: @f_with_convergent(
; CHECK: call i32 @llvm.convergent
; CHECK-NOT: call i32 @llvm.convergent

; ANALYSIS: for.body:
; ANALYSIS: Report: cannot add control dependency to convergent operation
define void @f_with_convergent() #1 {
entry:
  %a = load i32*, i32** @A, align 8
  %b = load i32*, i32** @B, align 8
  %c = load i32*, i32** @C, align 8
  %d = load i32*, i32** @D, align 8
  %e = load i32*, i32** @E, align 8
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %ind = phi i64 [ 0, %entry ], [ %add, %for.body ]

  %arrayidxA = getelementptr inbounds i32, i32* %a, i64 %ind
  %loadA = load i32, i32* %arrayidxA, align 4

  %arrayidxB = getelementptr inbounds i32, i32* %b, i64 %ind
  %loadB = load i32, i32* %arrayidxB, align 4

  %mulA = mul i32 %loadB, %loadA

  %add = add nuw nsw i64 %ind, 1
  %arrayidxA_plus_4 = getelementptr inbounds i32, i32* %a, i64 %add
  store i32 %mulA, i32* %arrayidxA_plus_4, align 4

  %arrayidxD = getelementptr inbounds i32, i32* %d, i64 %ind
  %loadD = load i32, i32* %arrayidxD, align 4

  %arrayidxE = getelementptr inbounds i32, i32* %e, i64 %ind
  %loadE = load i32, i32* %arrayidxE, align 4

  %convergentD = call i32 @llvm.convergent(i32 %loadD)
  %mulC = mul i32 %convergentD, %loadE

  %arrayidxC = getelementptr inbounds i32, i32* %c, i64 %ind
  store i32 %mulC, i32* %arrayidxC, align 4

  %exitcond = icmp eq i64 %add, 20
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; Make sure an explicit request for distribution is ignored if it
; requires possibly illegal checks.

; CHECK-LABEL: @f_with_convergent_forced_distribute(
; CHECK: call i32 @llvm.convergent
; CHECK-NOT: call i32 @llvm.convergent
define void @f_with_convergent_forced_distribute() #1 {
entry:
  %a = load i32*, i32** @A, align 8
  %b = load i32*, i32** @B, align 8
  %c = load i32*, i32** @C, align 8
  %d = load i32*, i32** @D, align 8
  %e = load i32*, i32** @E, align 8
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %ind = phi i64 [ 0, %entry ], [ %add, %for.body ]

  %arrayidxA = getelementptr inbounds i32, i32* %a, i64 %ind
  %loadA = load i32, i32* %arrayidxA, align 4

  %arrayidxB = getelementptr inbounds i32, i32* %b, i64 %ind
  %loadB = load i32, i32* %arrayidxB, align 4

  %mulA = mul i32 %loadB, %loadA

  %add = add nuw nsw i64 %ind, 1
  %arrayidxA_plus_4 = getelementptr inbounds i32, i32* %a, i64 %add
  store i32 %mulA, i32* %arrayidxA_plus_4, align 4

  %arrayidxD = getelementptr inbounds i32, i32* %d, i64 %ind
  %loadD = load i32, i32* %arrayidxD, align 4

  %arrayidxE = getelementptr inbounds i32, i32* %e, i64 %ind
  %loadE = load i32, i32* %arrayidxE, align 4

  %convergentD = call i32 @llvm.convergent(i32 %loadD)
  %mulC = mul i32 %convergentD, %loadE

  %arrayidxC = getelementptr inbounds i32, i32* %c, i64 %ind
  store i32 %mulC, i32* %arrayidxC, align 4

  %exitcond = icmp eq i64 %add, 20
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !0

for.end:                                          ; preds = %for.body
  ret void
}

attributes #0 = { nounwind readnone convergent }
attributes #1 = { nounwind convergent }

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.distribute.enable", i1 true}
