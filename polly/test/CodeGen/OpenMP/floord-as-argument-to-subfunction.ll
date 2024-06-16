; RUN: opt %loadNPMPolly -passes=polly-opt-isl -polly-opt-max-coefficient=-1 -polly-parallel -passes=polly-codegen -S < %s | FileCheck %s
;
; Check that we do not crash but generate parallel code
;
; CHECK: polly.par.setup
;
; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @III_hybrid(ptr %tsOut) #0 {
entry:
  br label %if.end

if.end:                                           ; preds = %entry
  br i1 undef, label %for.body42, label %for.cond66.preheader

for.cond39.for.cond66.preheader.loopexit67_crit_edge: ; preds = %for.body42
  %add.ptr62.lcssa = phi ptr [ undef, %for.body42 ]
  br label %for.cond66.preheader

for.cond66.preheader:                             ; preds = %for.cond39.for.cond66.preheader.loopexit67_crit_edge, %if.end
  %rawout1.3.ph = phi ptr [ %add.ptr62.lcssa, %for.cond39.for.cond66.preheader.loopexit67_crit_edge ], [ undef, %if.end ]
  %sb.3.ph = phi i32 [ 0, %for.cond39.for.cond66.preheader.loopexit67_crit_edge ], [ 0, %if.end ]
  %tspnt.3.ph = phi ptr [ undef, %for.cond39.for.cond66.preheader.loopexit67_crit_edge ], [ %tsOut, %if.end ]
  br label %for.cond69.preheader

for.body42:                                       ; preds = %if.end
  br label %for.cond39.for.cond66.preheader.loopexit67_crit_edge

for.cond69.preheader:                             ; preds = %for.end76, %for.cond66.preheader
  %tspnt.375 = phi ptr [ %incdec.ptr79, %for.end76 ], [ %tspnt.3.ph, %for.cond66.preheader ]
  %sb.374 = phi i32 [ %inc78, %for.end76 ], [ %sb.3.ph, %for.cond66.preheader ]
  %rawout1.373 = phi ptr [ undef, %for.end76 ], [ %rawout1.3.ph, %for.cond66.preheader ]
  br label %for.body71

for.body71:                                       ; preds = %for.body71, %for.cond69.preheader
  %indvars.iv = phi i64 [ 0, %for.cond69.preheader ], [ %indvars.iv.next, %for.body71 ]
  %rawout1.469 = phi ptr [ %rawout1.373, %for.cond69.preheader ], [ undef, %for.body71 ]
  %0 = load i64, ptr %rawout1.469, align 8
  %1 = shl nsw i64 %indvars.iv, 5
  %arrayidx73 = getelementptr inbounds double, ptr %tspnt.375, i64 %1
  store i64 %0, ptr %arrayidx73, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 18
  br i1 %exitcond, label %for.body71, label %for.end76

for.end76:                                        ; preds = %for.body71
  %inc78 = add nsw i32 %sb.374, 1
  %incdec.ptr79 = getelementptr inbounds double, ptr %tspnt.375, i64 1
  %exitcond95 = icmp ne i32 %inc78, 32
  br i1 %exitcond95, label %for.cond69.preheader, label %for.end80

for.end80:                                        ; preds = %for.end76
  ret void
}
