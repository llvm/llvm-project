; RUN: llc -verify-machineinstrs < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind
define void @foo(double* nocapture %x, double* nocapture readonly %y) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %y, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %add = fadd double %0, 1.000000e+00
  %arrayidx2 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  store double %add, double* %arrayidx2, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond19 = icmp eq i64 %indvars.iv.next, 1600
  br i1 %exitcond19, label %for.body7, label %for.body

; CHECK-LABEL: @foo

; CHECK-DAG: lfdu [[REG1:[0-9]+]], 8({{[0-9]+}})
; CHECK-DAG: fadd [[REG2:[0-9]+]], [[REG1]], 0
; CHECK-DAG: stfdu [[REG2]], 8({{[0-9]+}})
; CHECK: bdnz

; CHECK: blr

for.cond.cleanup6:                                ; preds = %for.body7
  ret void

for.body7:                                        ; preds = %for.body, %for.body7
  %i3.017 = phi i32 [ %inc9, %for.body7 ], [ 0, %for.body ]
  tail call void bitcast (void (...)* @bar to void ()*)() #0
  %inc9 = add nuw nsw i32 %i3.017, 1
  %exitcond = icmp eq i32 %inc9, 1024
  br i1 %exitcond, label %for.cond.cleanup6, label %for.body7
}

declare void @bar(...) 

attributes #0 = { nounwind }

