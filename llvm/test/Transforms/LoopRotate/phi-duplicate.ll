; RUN: opt -S -loop-rotate < %s | FileCheck %s
; RUN: opt -S -loop-rotate -enable-mssa-loop-dependency=true -verify-memoryssa < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0"

; PR5837
define void @test(i32 %N, double* %G) nounwind ssp {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %j.0 = phi i64 [ 1, %entry ], [ %inc, %for.body ] ; <i64> [#uses=5]
  %cmp = icmp slt i64 %j.0, 1000                  ; <i1> [#uses=1]
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds double, double* %G, i64 %j.0 ; <double*> [#uses=1]
  %tmp3 = load double, double* %arrayidx                  ; <double> [#uses=1]
  %sub = sub i64 %j.0, 1                          ; <i64> [#uses=1]
  %arrayidx6 = getelementptr inbounds double, double* %G, i64 %sub ; <double*> [#uses=1]
  %tmp7 = load double, double* %arrayidx6                 ; <double> [#uses=1]
  %add = fadd double %tmp3, %tmp7                 ; <double> [#uses=1]
  %arrayidx10 = getelementptr inbounds double, double* %G, i64 %j.0 ; <double*> [#uses=1]
  store double %add, double* %arrayidx10
  %inc = add nsw i64 %j.0, 1                      ; <i64> [#uses=1]
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; Should only end up with one phi.
; CHECK-LABEL:      define void @test(
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %for.body
; CHECK:      for.body:
; CHECK-NEXT:   %j.01 = phi i64
; CHECK-NOT:  br
; CHECK:   br i1 %cmp, label %for.body, label %for.end
; CHECK:      for.end:
; CHECK-NEXT:        ret void
