; RUN: opt %loadNPMPolly -aa-pipeline=basic-aa,scoped-noalias-aa,tbaa '-passes=print<polly-function-scops>' -disable-output < %s
;
; Ensure that ScopInfo's alias analysis llvm.memcpy for,
; like the AliasSetTracker, preserves bitcasts.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@tonemasks = external global [17 x [6 x [56 x float]]], align 16

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i32, i1) #0

; Function Attrs: nounwind uwtable
define void @setup_tone_curves() #1 {
entry:
  %workc = alloca [17 x [8 x [56 x float]]], align 16
  br label %for.cond7.preheader

for.cond7.preheader:                              ; preds = %for.cond7.preheader, %entry
  %indvars.iv45 = phi i64 [ %indvars.iv.next46, %for.cond7.preheader ], [ 0, %entry ]
  %indvars.iv.next46 = add nuw nsw i64 %indvars.iv45, 1
  %exitcond48 = icmp ne i64 %indvars.iv.next46, 56
  br i1 %exitcond48, label %for.cond7.preheader, label %for.body36

for.body36:                                       ; preds = %for.body36, %for.cond7.preheader
  %indvars.iv49 = phi i64 [ %indvars.iv.next50, %for.body36 ], [ 0, %for.cond7.preheader ]
  %arraydecay47 = getelementptr inbounds [17 x [6 x [56 x float]]], ptr @tonemasks, i64 0, i64 0, i64 %indvars.iv49, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %workc, ptr %arraydecay47, i64 224, i32 16, i1 false)
  %indvars.iv.next50 = add nuw nsw i64 %indvars.iv49, 1
  br i1 false, label %for.body36, label %for.end50

for.end50:                                        ; preds = %for.body36
  call void @llvm.memcpy.p0.p0.i64(ptr %workc, ptr @tonemasks, i64 224, i32 16, i1 false)
  br label %for.body74

for.body74:                                       ; preds = %for.body74, %for.end50
  %indvars.iv53 = phi i64 [ %indvars.iv.next54, %for.body74 ], [ 0, %for.end50 ]
  %arrayidx99 = getelementptr inbounds [17 x [8 x [56 x float]]], ptr %workc, i64 0, i64 0, i64 0, i64 %indvars.iv53
  %0 = load float, ptr %arrayidx99, align 4
  store float undef, ptr %arrayidx99, align 4
  %indvars.iv.next54 = add nuw nsw i64 %indvars.iv53, 1
  %exitcond57 = icmp ne i64 %indvars.iv.next54, 56
  br i1 %exitcond57, label %for.body74, label %for.inc104

for.inc104:                                       ; preds = %for.body74
  ret void
}

attributes #0 = { argmemonly nounwind }
attributes #1 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 4.0.0 (trunk 285057) (llvm/trunk 285063)"}
