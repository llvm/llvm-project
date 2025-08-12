; RUN: opt -mcpu=cortex-m7 -mtriple=thumbv8.1m.main -passes=loop-unroll -S  %s -o - | FileCheck %s --check-prefix=NLOB
; RUN: opt -mcpu=cortex-m55 -mtriple=thumbv8.1m.main -passes=loop-unroll -S  %s -o - | FileCheck %s --check-prefix=LOB

; This test checks behaviour of loop unrolling on processors with low overhead branching available 

; NLOB-LABEL: for.body{{.*}}.prol:
; NLOB-COUNT-1:     fmul fast float 
; NLOB-LABEL: for.body{{.*}}.prol.1:
; NLOB-COUNT-1:     fmul fast float 
; NLOB-LABEL: for.body{{.*}}.prol.2:
; NLOB-COUNT-1:     fmul fast float 
; NLOB-LABEL: for.body{{.*}}:
; NLOB-COUNT-4:     fmul fast float 
; NLOB-NOT:     fmul fast float 

; LOB-LABEL: for.body{{.*}}:
; LOB:     fmul fast float 
; LOB-NOT:     fmul fast float 


; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define dso_local void @test(i32 noundef %n, ptr nocapture noundef %pA) local_unnamed_addr #0 {
entry:
  %cmp46 = icmp sgt i32 %n, 0
  br i1 %cmp46, label %for.body, label %for.cond.cleanup

for.cond.loopexit:                                ; preds = %for.cond6.for.cond.cleanup8_crit_edge.us, %for.body
  %exitcond49.not = icmp eq i32 %add, %n
  br i1 %exitcond49.not, label %for.cond.cleanup, label %for.body, !llvm.loop !0

for.cond.cleanup:                                 ; preds = %for.cond.loopexit, %entry
  ret void

for.body:                                         ; preds = %entry, %for.cond.loopexit
  %k.047 = phi i32 [ %add, %for.cond.loopexit ], [ 0, %entry ]
  %add = add nuw nsw i32 %k.047, 1
  %cmp244 = icmp slt i32 %add, %n
  br i1 %cmp244, label %for.cond6.preheader.lr.ph, label %for.cond.loopexit

for.cond6.preheader.lr.ph:                        ; preds = %for.body
  %invariant.gep = getelementptr float, ptr %pA, i32 %k.047
  br label %for.cond6.preheader.us

for.cond6.preheader.us:                           ; preds = %for.cond6.for.cond.cleanup8_crit_edge.us, %for.cond6.preheader.lr.ph
  %w.045.us = phi i32 [ %add, %for.cond6.preheader.lr.ph ], [ %inc19.us, %for.cond6.for.cond.cleanup8_crit_edge.us ]
  %mul.us = mul nuw nsw i32 %w.045.us, %n
  %0 = getelementptr float, ptr %pA, i32 %mul.us
  %arrayidx.us = getelementptr float, ptr %0, i32 %k.047
  br label %for.body9.us

for.body9.us:                                     ; preds = %for.cond6.preheader.us, %for.body9.us
  %x.043.us = phi i32 [ %add, %for.cond6.preheader.us ], [ %inc.us, %for.body9.us ]
  %1 = load float, ptr %arrayidx.us, align 4
  %mul11.us = mul nuw nsw i32 %x.043.us, %n
  %gep.us = getelementptr float, ptr %invariant.gep, i32 %mul11.us
  %2 = load float, ptr %gep.us, align 4
  %mul14.us = fmul fast float %2, %1
  %arrayidx17.us = getelementptr float, ptr %0, i32 %x.043.us
  store float %mul14.us, ptr %arrayidx17.us, align 4
  %inc.us = add nuw nsw i32 %x.043.us, 1
  %exitcond.not = icmp eq i32 %inc.us, %n
  br i1 %exitcond.not, label %for.cond6.for.cond.cleanup8_crit_edge.us, label %for.body9.us

for.cond6.for.cond.cleanup8_crit_edge.us:         ; preds = %for.body9.us
  %inc19.us = add nuw nsw i32 %w.045.us, 1
  %exitcond48.not = icmp eq i32 %inc19.us, %n
  br i1 %exitcond48.not, label %for.cond.loopexit, label %for.cond6.preheader.us
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.unroll.disable"}
