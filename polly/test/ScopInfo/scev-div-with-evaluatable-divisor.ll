; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s

; Derived from test-suite/SingleSource/UnitTests/Vector/SSE/sse.stepfft.c

; The values %mul.i44 is simplified to constant 4 by ScalarEvolution, but 
; SCEVAffinator used to check whether the sdiv's argument was constant.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define void @cfft2(i32 %n, double* %A) local_unnamed_addr #0 {
entry:
  br i1 true, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.inc, %for.body.lr.ph
  %mj.017 = phi i32 [ 1, %for.body.lr.ph ], [ undef, %for.inc ]
  br i1 true, label %if.else, label %if.then

if.then:                                          ; preds = %for.body
  br label %for.inc

if.else:                                          ; preds = %for.body
  %mul.i44 = shl i32 %mj.017, 2
  %div.i45 = sdiv i32 %n, %mul.i44
  br i1 true, label %for.body.i58.lr.ph, label %for.inc

for.body.i58.lr.ph:                               ; preds = %if.else
  br i1 false, label %for.body.i58.us, label %for.body.i58.preheader

for.body.i58.preheader:                           ; preds = %for.body.i58.lr.ph
  br label %for.body.i58

for.body.i58.us:                                  ; preds = %for.body.i58.us, %for.body.i58.lr.ph
  br i1 false, label %for.inc, label %for.body.i58.us

for.body.i58:                                     ; preds = %for.body.i58, %for.body.i58.preheader
  store double 0.0, double* %A
  %exitcond42 = icmp eq i32 0, %div.i45
  br i1 %exitcond42, label %for.inc, label %for.body.i58

for.inc:                                          ; preds = %for.body.i58, %for.body.i58.us, %if.else, %if.then
  br i1 undef, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry
  ret void
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.9.0 (trunk 273249) (llvm/trunk 273255)"}

; CHECK-LABEL: Stmt_for_body_i58
; CHECK-NEXT:      Domain :=
; CHECK-NEXT:          [n] -> { Stmt_for_body_i58[0] : -3 <= n <= 3 };
