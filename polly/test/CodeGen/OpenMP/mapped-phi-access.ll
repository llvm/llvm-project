; RUN: opt %loadNPMPolly -polly-stmt-granularity=bb -polly-parallel '-passes=polly-delicm,polly-codegen' -S < %s | FileCheck %s
;
; Verify that -polly-parallel can handle mapped scalar MemoryAccesses.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @main() local_unnamed_addr #0 {
entry:
  %0 = load ptr, ptr undef, align 8, !tbaa !1
  %1 = load ptr, ptr undef, align 8, !tbaa !1
  br label %for.body65.i226

for.body65.i226:                                  ; preds = %for.inc85.i238, %entry
  %indvars.iv8.i223 = phi i64 [ 0, %entry ], [ %indvars.iv.next9.i236, %for.inc85.i238 ]
  %arrayidx70.i224 = getelementptr inbounds double, ptr %1, i64 %indvars.iv8.i223
  br label %for.body68.i235

for.body68.i235:                                  ; preds = %for.body68.i235, %for.body65.i226
  %2 = phi double [ undef, %for.body65.i226 ], [ undef, %for.body68.i235 ]
  %indvars.iv.i227 = phi i64 [ 0, %for.body65.i226 ], [ %indvars.iv.next.i233, %for.body68.i235 ]
  %arrayidx74.i228 = getelementptr inbounds [4000 x double], ptr %0, i64 %indvars.iv8.i223, i64 %indvars.iv.i227
  %3 = load double, ptr %arrayidx74.i228, align 8, !tbaa !5
  store double undef, ptr %arrayidx70.i224, align 8, !tbaa !5
  %indvars.iv.next.i233 = add nuw nsw i64 %indvars.iv.i227, 1
  %exitcond.i234 = icmp eq i64 %indvars.iv.next.i233, 4000
  br i1 %exitcond.i234, label %for.inc85.i238, label %for.body68.i235

for.inc85.i238:                                   ; preds = %for.body68.i235
  %indvars.iv.next9.i236 = add nuw nsw i64 %indvars.iv8.i223, 1
  %exitcond10.i237 = icmp eq i64 %indvars.iv.next9.i236, 4000
  br i1 %exitcond10.i237, label %kernel_gemver_StrictFP.exit, label %for.body65.i226

kernel_gemver_StrictFP.exit:                      ; preds = %for.inc85.i238
  ret void
}

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 6.0.0 "}
!1 = !{!2, !2, i64 0}
!2 = !{!"any pointer", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!6, !6, i64 0}
!6 = !{!"double", !3, i64 0}


; CHECK-LABEL: define internal void @main_polly_subfn(ptr %polly.par.userContext)
;
; CHECK:       polly.stmt.for.body65.i226:
; CHECK-NEXT:    %polly.access.polly.subfunc.arg.[[R1:[0-9]*]] = getelementptr double, ptr %polly.subfunc.arg.[[R0:[0-9]*]], i64 %polly.indvar
; CHECK-NEXT:    store double undef, ptr %polly.access.polly.subfunc.arg.[[R1]]
