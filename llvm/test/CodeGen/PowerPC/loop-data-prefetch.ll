; RUN: llc -verify-machineinstrs -enable-ppc-prefetching=true -mcpu=a2 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux"

; Function Attrs: nounwind
define void @foo(ptr nocapture %a, ptr nocapture readonly %b) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %add = fadd double %0, 1.000000e+00
  %arrayidx2 = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  store double %add, ptr %arrayidx2, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1600
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void

; CHECK-LABEL: @foo
; CHECK: dcbt
}

attributes #0 = { nounwind }

