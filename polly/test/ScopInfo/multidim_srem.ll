; RUN: opt %loadPolly -analyze -polly-scops -S < %s | FileCheck %s
;
;    void foo(long n, float A[][n][n]) {
;      for (long i = 0; i < 200; i++)
;        for (long j = 0; j < n; j++)
;          for (long k = 0; k < n; k++)
;            A[i % 2][j][k] += 10;
;    }

; CHECK:      Statements {
; CHECK-NEXT:     Stmt_for_body_8
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [n] -> { Stmt_for_body_8[i0, i1, i2] : 0 <= i0 <= 199 and 0 <= i1 < n and 0 <= i2 < n };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [n] -> { Stmt_for_body_8[i0, i1, i2] -> [i0, i1, i2] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt_for_body_8[i0, i1, i2] -> MemRef_A[o0, i1, i2] : 2*floor((i0 + o0)/2) = i0 + o0 and 0 <= o0 <= 1 };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt_for_body_8[i0, i1, i2] -> MemRef_A[o0, i1, i2] : 2*floor((i0 + o0)/2) = i0 + o0 and 0 <= o0 <= 1 };
; CHECK-NEXT: }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"


define void @foo(i64 %n, float* %A) #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.cond.1.preheader

for.cond.1.preheader:                             ; preds = %entry.split, %for.inc.14
  %i.06 = phi i64 [ 0, %entry.split ], [ %inc15, %for.inc.14 ]
  %cmp2.3 = icmp sgt i64 %n, 0
  br i1 %cmp2.3, label %for.cond.5.preheader.lr.ph, label %for.inc.14

for.cond.5.preheader.lr.ph:                       ; preds = %for.cond.1.preheader
  br label %for.cond.5.preheader

for.cond.5.preheader:                             ; preds = %for.cond.5.preheader.lr.ph, %for.inc.11
  %j.04 = phi i64 [ 0, %for.cond.5.preheader.lr.ph ], [ %inc12, %for.inc.11 ]
  %cmp6.1 = icmp sgt i64 %n, 0
  br i1 %cmp6.1, label %for.body.8.lr.ph, label %for.inc.11

for.body.8.lr.ph:                                 ; preds = %for.cond.5.preheader
  br label %for.body.8

for.body.8:                                       ; preds = %for.body.8.lr.ph, %for.body.8
  %k.02 = phi i64 [ 0, %for.body.8.lr.ph ], [ %inc, %for.body.8 ]
  %rem = srem i64 %i.06, 2
  %0 = mul nuw i64 %n, %n
  %1 = mul nsw i64 %0, %rem
  %arrayidx = getelementptr inbounds float, float* %A, i64 %1
  %2 = mul nsw i64 %j.04, %n
  %arrayidx9 = getelementptr inbounds float, float* %arrayidx, i64 %2
  %arrayidx10 = getelementptr inbounds float, float* %arrayidx9, i64 %k.02
  %3 = load float, float* %arrayidx10, align 4, !tbaa !1
  %add = fadd float %3, 1.000000e+01
  store float %add, float* %arrayidx10, align 4, !tbaa !1
  %inc = add nuw nsw i64 %k.02, 1
  %exitcond = icmp ne i64 %inc, %n
  br i1 %exitcond, label %for.body.8, label %for.cond.5.for.inc.11_crit_edge

for.cond.5.for.inc.11_crit_edge:                  ; preds = %for.body.8
  br label %for.inc.11

for.inc.11:                                       ; preds = %for.cond.5.for.inc.11_crit_edge, %for.cond.5.preheader
  %inc12 = add nuw nsw i64 %j.04, 1
  %exitcond7 = icmp ne i64 %inc12, %n
  br i1 %exitcond7, label %for.cond.5.preheader, label %for.cond.1.for.inc.14_crit_edge

for.cond.1.for.inc.14_crit_edge:                  ; preds = %for.inc.11
  br label %for.inc.14

for.inc.14:                                       ; preds = %for.cond.1.for.inc.14_crit_edge, %for.cond.1.preheader
  %inc15 = add nuw nsw i64 %i.06, 1
  %exitcond8 = icmp ne i64 %inc15, 200
  br i1 %exitcond8, label %for.cond.1.preheader, label %for.end.16

for.end.16:                                       ; preds = %for.inc.14
  ret void
}

; Function Attrs: nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture) #1

; Function Attrs: nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) #1

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.7.0 (trunk 240923) (llvm/trunk 240924)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"float", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
