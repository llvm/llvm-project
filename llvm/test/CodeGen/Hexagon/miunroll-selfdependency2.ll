; RUN: llc -O3 -march=hexagon -debug-only=machine-unroller < %s 2>&1 |\
; RUN:  FileCheck %s

; CHECK: Using unroll factor of 2

%struct.loop_params_s = type { i32, ptr, i32 }

define dso_local float @foo(ptr nocapture readonly %in) local_unnamed_addr {
entry:
  %0 = load i32, ptr %in, align 4
  %n1 = getelementptr inbounds %struct.loop_params_s, ptr %in, i32 0, i32 2
  %1 = load i32, ptr %n1, align 4
  %v = getelementptr inbounds %struct.loop_params_s, ptr %in, i32 0, i32 1
  %2 = load ptr, ptr %v, align 4
  %3 = load ptr, ptr %2, align 4
  %cmp31 = icmp sgt i32 %0, 0
  %cmp528 = icmp sgt i32 %1, 0
  %or.cond = and i1 %cmp31, %cmp528
  br i1 %or.cond, label %for.cond4.preheader.us.preheader, label %for.end12

for.cond4.preheader.us.preheader:                 ; preds = %entry
  %arrayidx3 = getelementptr inbounds ptr, ptr %2, i32 1
  %4 = load ptr, ptr %arrayidx3, align 4
  br label %for.cond4.preheader.us

for.cond4.preheader.us:                           ; preds = %for.cond4.for.inc10_crit_edge.us, %for.cond4.preheader.us.preheader
  %q.033.us = phi float [ %add9.us, %for.cond4.for.inc10_crit_edge.us ], [ 0.000000e+00, %for.cond4.preheader.us.preheader ]
  %i.032.us = phi i32 [ %inc11.us, %for.cond4.for.inc10_crit_edge.us ], [ 0, %for.cond4.preheader.us.preheader ]
  br label %for.body6.us

for.body6.us:                                     ; preds = %for.body6.us, %for.cond4.preheader.us
  %q.130.us = phi float [ %q.033.us, %for.cond4.preheader.us ], [ %add9.us, %for.body6.us ]
  %.pn = phi ptr [ %4, %for.cond4.preheader.us ], [ %arrayidx7.us.phi, %for.body6.us ]
  %arrayidx8.us.phi = phi ptr [ %3, %for.cond4.preheader.us ], [ %arrayidx8.us.inc, %for.body6.us ]
  %k.029.us = phi i32 [ 0, %for.cond4.preheader.us ], [ %add.us, %for.body6.us ]
  %arrayidx7.us.phi = getelementptr float, ptr %.pn, i32 1
  %add.us = add nuw nsw i32 %k.029.us, 1
  %5 = load float, ptr %arrayidx7.us.phi, align 4
  %6 = load float, ptr %arrayidx8.us.phi, align 4
  %mul.us = fmul float %5, %6
  %add9.us = fadd float %q.130.us, %mul.us
  %exitcond = icmp eq i32 %add.us, %1
  %arrayidx8.us.inc = getelementptr float, ptr %arrayidx8.us.phi, i32 1
  br i1 %exitcond, label %for.cond4.for.inc10_crit_edge.us, label %for.body6.us

for.cond4.for.inc10_crit_edge.us:                 ; preds = %for.body6.us
  %inc11.us = add nuw nsw i32 %i.032.us, 1
  %exitcond36 = icmp eq i32 %inc11.us, %0
  br i1 %exitcond36, label %for.end12, label %for.cond4.preheader.us

for.end12:                                        ; preds = %for.cond4.for.inc10_crit_edge.us, %entry
  %q.0.lcssa = phi float [ 0.000000e+00, %entry ], [ %add9.us, %for.cond4.for.inc10_crit_edge.us ]
  ret float %q.0.lcssa
}
