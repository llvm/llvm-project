; RUN: opt %loadPolly -polly-opt-isl -polly-opt-max-coefficient=-1 -polly-parallel -polly-codegen -S < %s | FileCheck %s
;
; Check that we do not crash but generate parallel code
;
; CHECK: polly.par.setup
;
; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @III_hybrid([32 x double]* %tsOut) #0 {
entry:
  %0 = getelementptr inbounds [32 x double], [32 x double]* %tsOut, i64 0, i64 0
  br label %if.end

if.end:                                           ; preds = %entry
  br i1 undef, label %for.body42, label %for.cond66.preheader

for.cond39.for.cond66.preheader.loopexit67_crit_edge: ; preds = %for.body42
  %add.ptr62.lcssa = phi double* [ undef, %for.body42 ]
  br label %for.cond66.preheader

for.cond66.preheader:                             ; preds = %for.cond39.for.cond66.preheader.loopexit67_crit_edge, %if.end
  %rawout1.3.ph = phi double* [ %add.ptr62.lcssa, %for.cond39.for.cond66.preheader.loopexit67_crit_edge ], [ undef, %if.end ]
  %sb.3.ph = phi i32 [ 0, %for.cond39.for.cond66.preheader.loopexit67_crit_edge ], [ 0, %if.end ]
  %tspnt.3.ph = phi double* [ undef, %for.cond39.for.cond66.preheader.loopexit67_crit_edge ], [ %0, %if.end ]
  br label %for.cond69.preheader

for.body42:                                       ; preds = %if.end
  br label %for.cond39.for.cond66.preheader.loopexit67_crit_edge

for.cond69.preheader:                             ; preds = %for.end76, %for.cond66.preheader
  %tspnt.375 = phi double* [ %incdec.ptr79, %for.end76 ], [ %tspnt.3.ph, %for.cond66.preheader ]
  %sb.374 = phi i32 [ %inc78, %for.end76 ], [ %sb.3.ph, %for.cond66.preheader ]
  %rawout1.373 = phi double* [ undef, %for.end76 ], [ %rawout1.3.ph, %for.cond66.preheader ]
  br label %for.body71

for.body71:                                       ; preds = %for.body71, %for.cond69.preheader
  %indvars.iv = phi i64 [ 0, %for.cond69.preheader ], [ %indvars.iv.next, %for.body71 ]
  %rawout1.469 = phi double* [ %rawout1.373, %for.cond69.preheader ], [ undef, %for.body71 ]
  %1 = bitcast double* %rawout1.469 to i64*
  %2 = load i64, i64* %1, align 8
  %3 = shl nsw i64 %indvars.iv, 5
  %arrayidx73 = getelementptr inbounds double, double* %tspnt.375, i64 %3
  %4 = bitcast double* %arrayidx73 to i64*
  store i64 %2, i64* %4, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 18
  br i1 %exitcond, label %for.body71, label %for.end76

for.end76:                                        ; preds = %for.body71
  %inc78 = add nsw i32 %sb.374, 1
  %incdec.ptr79 = getelementptr inbounds double, double* %tspnt.375, i64 1
  %exitcond95 = icmp ne i32 %inc78, 32
  br i1 %exitcond95, label %for.cond69.preheader, label %for.end80

for.end80:                                        ; preds = %for.end76
  ret void
}
