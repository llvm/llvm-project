; RUN: llc < %s -march=ppc64 | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind
define hidden void @_mpd_shortdiv(i64 %n) #0 {
entry:
  br i1 undef, label %for.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %i.018.in = phi i64 [ %n, %for.body.lr.ph ], [ %i.018, %for.body ]
  %i.018 = add i64 %i.018.in, -1
  %add.i = or i128 undef, undef
  %div.i = udiv i128 %add.i, 0
  %conv3.i11 = trunc i128 %div.i to i64
  store i64 %conv3.i11, i64* undef, align 8
  %cmp = icmp eq i64 %i.018, 0
  br i1 %cmp, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; CHECK-LABEL: @_mpd_shortdiv
; CHECK-NOT: mtctr

attributes #0 = { nounwind }

