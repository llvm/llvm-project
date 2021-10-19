; RUN: opt %loadPolly -polly-codegen -S < %s
;
; The entry of this scop's simple region (entry.split => for.end) has an trivial
; PHI node. LCSSA may create such PHI nodes. This is a breakdown of this case in
; the function 'Laguerre_With_Deflation' of oggenc from LLVM's test-suite.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @test(i64 %n, float* noalias nonnull %A, float %a) {
entry:
  br label %entry.split

; CHECK-LABEL: %polly.split_new_and_old
; CHECK-NEXT:    store float %a, float* %b.phiops

; CHECK-LABEL: polly.stmt.entry.split
; CHECK-NEXT:    %b.phiops.reload = load float, float* %b.phiops

entry.split:
  %b = phi float [ %a, %entry ]
  store float %b, float* %A, align 4
  %cmp2 = icmp slt i64 %n, 5
  br i1 %cmp2, label %for.cond, label %for.end

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i64 [ 0, %entry.split ], [ %add, %for.inc ]
  %cmp = icmp slt i64 %i.0, %n
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds float, float* %A, i64 %i.0
  store float %a, float* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %add = add nuw nsw i64 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
