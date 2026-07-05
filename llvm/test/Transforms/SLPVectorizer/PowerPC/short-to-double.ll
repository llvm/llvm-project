; RUN: opt -S -mtriple=powerpc64-linux-gnu -mcpu=pwr9 -mattr=+vsx -passes=slp-vectorizer < %s | FileCheck %s --check-prefix=CHECK-P9
; RUN: opt -S -mtriple=powerpc64-linux-gnu -mcpu=pwr8 -mattr=+vsx -passes=slp-vectorizer < %s | FileCheck %s --check-prefix=CHECK-P8

%struct._pp = type { i16, i16, i16, i16 }

; Function Attrs: norecurse nounwind readonly
define [5 x double] @foo(double %k, i64 %n, ptr nocapture readonly %p) local_unnamed_addr #0 {
entry:
  %cmp17 = icmp sgt i64 %n, 0
  br i1 %cmp17, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %retval.sroa.0.0.lcssa = phi double [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %retval.sroa.4.0.lcssa = phi double [ 0.000000e+00, %entry ], [ %add10, %for.body ]
  %.fca.0.insert = insertvalue [5 x double] undef, double %retval.sroa.0.0.lcssa, 0
  %.fca.1.insert = insertvalue [5 x double] %.fca.0.insert, double %retval.sroa.4.0.lcssa, 1
  ret [5 x double] %.fca.1.insert

for.body:                                         ; preds = %entry, %for.body
  %i.020 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %retval.sroa.4.019 = phi double [ %add10, %for.body ], [ 0.000000e+00, %entry ]
  %retval.sroa.0.018 = phi double [ %add, %for.body ], [ 0.000000e+00, %entry ]
  %r1 = getelementptr inbounds %struct._pp, ptr %p, i64 %i.020, i32 2
  %0 = load i16, ptr %r1, align 2
  %conv2 = uitofp i16 %0 to double
  %mul = fmul double %conv2, %k
  %add = fadd double %retval.sroa.0.018, %mul
  %g5 = getelementptr inbounds %struct._pp, ptr %p, i64 %i.020, i32 1
  %1 = load i16, ptr %g5, align 2
  %conv7 = uitofp i16 %1 to double
  %mul8 = fmul double %conv7, %k
  %add10 = fadd double %retval.sroa.4.019, %mul8
  %inc = add nuw nsw i64 %i.020, 1
  %exitcond = icmp eq i64 %inc, %n
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; CHECK-P8: load <2 x i16>
; CHECK-P9-NOT: load <2 x i16>
