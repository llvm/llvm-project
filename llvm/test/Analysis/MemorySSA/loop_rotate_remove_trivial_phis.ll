; RUN: opt -loop-rotate -print-memoryssa -disable-output -enable-mssa-loop-dependency -verify-memoryssa %s 2>&1 |  FileCheck %s
; REQUIRES: asserts

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

declare double @sqrt(double)

; CHECK-LABEL: @f
define internal fastcc double @f(i32* %n_, double* %dx) align 32 {
entry:
; CHECK: entry:
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NOT: 7 = MemoryPhi
; CHECK-NOT: 6 = MemoryPhi
  %v0 = load i32, i32* %n_, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %xmax.0 = phi double [ undef, %entry ], [ %xmax.1, %for.body ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp slt i32 %i.0, %v0
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %idxprom = zext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds double, double* %dx, i64 %idxprom
  %v1 = load double, double* %arrayidx, align 8
  %cmp1 = fcmp ueq double %v1, 0.000000e+00
  %xmax.1 = select i1 %cmp1, double %xmax.0, double %v1
  %inc = add nuw nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %xmax.0.lcssa = phi double [ %xmax.0, %for.cond ]
  %cmp2 = fcmp oeq double %xmax.0.lcssa, 0.000000e+00
  br i1 %cmp2, label %cleanup, label %if.end4

if.end4:                                          ; preds = %for.end
  %div = fdiv double 1.000000e+00, %xmax.0.lcssa
  %cmp61 = icmp slt i32 0, %v0
  br i1 %cmp61, label %for.body7.lr.ph, label %for.end15

for.body7.lr.ph:                                  ; preds = %if.end4
  br label %for.body7

; CHECK: for.body7:
; CHECK: 3 = MemoryPhi({for.body7.lr.ph,liveOnEntry},{for.body7,1})
for.body7:                                        ; preds = %for.body7.lr.ph, %for.body7
  %i.13 = phi i32 [ 0, %for.body7.lr.ph ], [ %inc14, %for.body7 ]
  %sum.02 = phi x86_fp80 [ undef, %for.body7.lr.ph ], [ %add, %for.body7 ]
  %idxprom9 = zext i32 %i.13 to i64
  %arrayidx10 = getelementptr inbounds double, double* %dx, i64 %idxprom9
  %v3 = load double, double* %arrayidx10, align 8
  %mul11 = fmul double %div, %v3
  %v2 = call double @sqrt(double %v3)
  %mul12 = fmul double %mul11, %v2
  %conv = fpext double %mul12 to x86_fp80
  %add = fadd x86_fp80 %sum.02, %conv
  %inc14 = add nuw nsw i32 %i.13, 1
  %cmp6 = icmp slt i32 %inc14, %v0
  br i1 %cmp6, label %for.body7, label %for.cond5.for.end15_crit_edge

for.cond5.for.end15_crit_edge:                    ; preds = %for.body7
  %split = phi x86_fp80 [ %add, %for.body7 ]
  br label %for.end15

for.end15:                                        ; preds = %for.cond5.for.end15_crit_edge, %if.end4
  %sum.0.lcssa = phi x86_fp80 [ %split, %for.cond5.for.end15_crit_edge ], [ undef, %if.end4 ]
  %conv16 = fptrunc x86_fp80 %sum.0.lcssa to double
  %call = call double @sqrt(double %conv16)
  %mul17 = fmul double %call, 0.000000e+00
  br label %cleanup

cleanup:                                          ; preds = %for.end15, %for.end
  %retval.0 = phi double [ undef, %for.end ], [ %mul17, %for.end15 ]
  ret double %retval.0
}
