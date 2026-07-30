; RUN: llc -enable-ppc-prefetching=true -verify-machineinstrs < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux"

; Function Attrs: nounwind
define void @foo(ptr %x, ptr nocapture readonly %y) #0 {
entry:
  %scevgep = getelementptr double, ptr %x, i64 1599
  %scevgep20 = getelementptr double, ptr %y, i64 1599
  br label %vector.memcheck

vector.memcheck:                                  ; preds = %for.end, %entry
  %j.015 = phi i32 [ 0, %entry ], [ %inc7, %for.end ]
  %bound0 = icmp uge ptr %scevgep20, %x
  %bound1 = icmp uge ptr %scevgep, %y
  %memcheck.conflict = and i1 %bound0, %bound1
  br i1 %memcheck.conflict, label %middle.block, label %vector.body

vector.body:                                      ; preds = %vector.memcheck, %vector.body
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %vector.memcheck ]
  %0 = getelementptr inbounds double, ptr %y, i64 %index
  %wide.load = load <4 x double>, ptr %0, align 8
  %1 = fadd <4 x double> %wide.load, <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>
  %2 = getelementptr inbounds double, ptr %x, i64 %index
  store <4 x double> %1, ptr %2, align 8
  %index.next = add i64 %index, 4
  %3 = icmp eq i64 %index.next, 1600
  br i1 %3, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body, %vector.memcheck
  %resume.val = phi i1 [ false, %vector.memcheck ], [ true, %vector.body ]
  %trunc.resume.val = phi i64 [ 0, %vector.memcheck ], [ 1600, %vector.body ]
  br i1 %resume.val, label %for.end, label %for.body3

for.body3:                                        ; preds = %middle.block, %for.body3
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body3 ], [ %trunc.resume.val, %middle.block ]
  %arrayidx = getelementptr inbounds double, ptr %y, i64 %indvars.iv
  %4 = load double, ptr %arrayidx, align 8
  %add = fadd double %4, 1.000000e+00
  %arrayidx5 = getelementptr inbounds double, ptr %x, i64 %indvars.iv
  store double %add, ptr %arrayidx5, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1600
  br i1 %exitcond, label %for.end, label %for.body3

for.end:                                          ; preds = %middle.block, %for.body3
  tail call void @bar(ptr %x) #2
  %inc7 = add nuw nsw i32 %j.015, 1
  %exitcond16 = icmp eq i32 %inc7, 100
  br i1 %exitcond16, label %for.end8, label %vector.memcheck

for.end8:                                         ; preds = %for.end
  ret void

; CHECK-LABEL: @foo
; CHECK: dcbt
}

declare void @bar(ptr) #1

attributes #0 = { nounwind "target-cpu"="a2q" }
attributes #1 = { "target-cpu"="a2q" }
attributes #2 = { nounwind }

