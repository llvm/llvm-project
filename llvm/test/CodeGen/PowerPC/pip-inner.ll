; RUN: llc -verify-machineinstrs < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind
define void @foo(double* %x, double* nocapture readonly %y) #0 {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.end, %entry
  %i.015 = phi i32 [ 0, %entry ], [ %inc7, %for.end ]
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %for.cond1.preheader
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.body3 ]
  %arrayidx = getelementptr inbounds double, double* %y, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %add = fadd double %0, 1.000000e+00
  %arrayidx5 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  store double %add, double* %arrayidx5, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 16000
  br i1 %exitcond, label %for.end, label %for.body3

for.end:                                          ; preds = %for.body3
  tail call void @bar(double* %x) #2
  %inc7 = add nuw nsw i32 %i.015, 1
  %exitcond16 = icmp eq i32 %inc7, 1000
  br i1 %exitcond16, label %for.end8, label %for.cond1.preheader

for.end8:                                         ; preds = %for.end
  ret void

; CHECK-LABEL: @foo

; CHECK: lfdu [[REG1:[0-9]+]], 8({{[0-9]+}})
; CHECK: fadd [[REG2:[0-9]+]], [[REG1]], {{[0-9]+}}
; CHECK: stfdu [[REG2]], 8({{[0-9]+}})
; CHECK: bdnz

; CHECK: bl bar
; CHECK-NEXT: nop

; CHECK: blr
}

declare void @bar(double*) #1

attributes #0 = { nounwind "target-cpu"="a2" }
attributes #1 = { "target-cpu"="a2" }
attributes #2 = { nounwind }

