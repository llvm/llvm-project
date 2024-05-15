; RUN: opt %loadPolly -polly-parallel \
; RUN: -polly-parallel-force -polly-codegen \
; RUN: -S -verify-dom-info < %s \
; RUN: | FileCheck %s -check-prefix=IR

; RUN: opt %loadPolly -polly-parallel \
; RUN: -polly-parallel-force -polly-codegen -polly-scheduling=runtime \
; RUN: -S -verify-dom-info < %s \
; RUN: | FileCheck %s -check-prefix=IR

; RUN: opt %loadPolly -polly-parallel \
; RUN: -polly-parallel-force -polly-codegen -polly-omp-backend=LLVM \
; RUN: -S -verify-dom-info < %s \
; RUN: | FileCheck %s -check-prefix=LIBOMP-IR

; IR: @GOMP_parallel_loop_runtime_start

; LIBOMP-IR: call void (ptr, i32, ptr, ...) @__kmpc_fork_call
; LIBOMP-IR: call void @__kmpc_dispatch_init_{{[4|8]}}

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@longLimit = external global [9 x [23 x i32]], align 16
@shortLimit = external global [9 x [14 x i32]], align 16

define void @init_layer3(i32 %down_sample_sblimit) #0 {
entry:
  br label %for.cond.463.preheader

for.cond.463.preheader:                           ; preds = %entry
  br label %for.cond.499.preheader

for.cond.533.preheader:                           ; preds = %for.inc.530
  ret void

for.cond.499.preheader:                           ; preds = %for.inc.530, %for.cond.463.preheader
  %indvars.iv140 = phi i64 [ 0, %for.cond.463.preheader ], [ %indvars.iv.next141, %for.inc.530 ]
  %arrayidx483 = getelementptr inbounds [9 x [23 x i32]], ptr @longLimit, i64 0, i64 %indvars.iv140, i64 0
  store i32 undef, ptr %arrayidx483, align 4, !tbaa !1
  %arrayidx487 = getelementptr inbounds [9 x [23 x i32]], ptr @longLimit, i64 0, i64 %indvars.iv140, i64 0
  %tmp = load i32, ptr %arrayidx487, align 4, !tbaa !1
  %indvars.iv.next135 = add nuw nsw i64 0, 1
  br label %for.body.502

for.body.502:                                     ; preds = %for.inc.527, %for.cond.499.preheader
  %indvars.iv137 = phi i64 [ 0, %for.cond.499.preheader ], [ %indvars.iv.next138, %for.inc.527 ]
  %arrayidx518 = getelementptr inbounds [9 x [14 x i32]], ptr @shortLimit, i64 0, i64 %indvars.iv140, i64 %indvars.iv137
  %tmp1 = load i32, ptr %arrayidx518, align 4, !tbaa !1
  %cmp519 = icmp sgt i32 %tmp1, %down_sample_sblimit
  br i1 %cmp519, label %if.then.521, label %for.inc.527

if.then.521:                                      ; preds = %for.body.502
  br label %for.inc.527

for.inc.527:                                      ; preds = %if.then.521, %for.body.502
  %indvars.iv.next138 = add nuw nsw i64 %indvars.iv137, 1
  %exitcond139 = icmp ne i64 %indvars.iv.next138, 14
  br i1 %exitcond139, label %for.body.502, label %for.inc.530

for.inc.530:                                      ; preds = %for.inc.527
  %indvars.iv.next141 = add nuw nsw i64 %indvars.iv140, 1
  %exitcond142 = icmp ne i64 %indvars.iv.next141, 9
  br i1 %exitcond142, label %for.cond.499.preheader, label %for.cond.533.preheader
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.8.0 (trunk 246359)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
