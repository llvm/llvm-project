; RUN: sed -e s/REPLACE_PSTATE_MACRO/aarch64_pstate_sm_enabled/ %s | opt -passes=loop-vectorize,slp-vectorizer -S - | FileCheck %s --check-prefix=CHECK
; RUN: sed -e s/REPLACE_PSTATE_MACRO/aarch64_pstate_sm_enabled/ %s | opt -passes=loop-vectorize,slp-vectorizer -S -enable-scalable-autovec-in-streaming-mode - | FileCheck %s --check-prefix=CHECK-FORCE-SCALABLE
; RUN: sed -e s/REPLACE_PSTATE_MACRO/aarch64_pstate_sm_enabled/ %s | opt -passes=loop-vectorize,slp-vectorizer -S -enable-fixedwidth-autovec-in-streaming-mode - | FileCheck %s --check-prefix=CHECK-FORCE-FIXEDWIDTH

; RUN: sed -e s/REPLACE_PSTATE_MACRO/aarch64_pstate_sm_compatible/ %s | opt -passes=loop-vectorize,slp-vectorizer -S - | FileCheck %s --check-prefix=CHECK
; RUN: sed -e s/REPLACE_PSTATE_MACRO/aarch64_pstate_sm_compatible/ %s | opt -passes=loop-vectorize,slp-vectorizer -S -enable-scalable-autovec-in-streaming-mode - | FileCheck %s --check-prefix=CHECK-FORCE-SCALABLE
; RUN: sed -e s/REPLACE_PSTATE_MACRO/aarch64_pstate_sm_compatible/ %s | opt -passes=loop-vectorize,slp-vectorizer -S -enable-fixedwidth-autovec-in-streaming-mode - | FileCheck %s --check-prefix=CHECK-FORCE-FIXEDWIDTH

; RUN: sed -e s/REPLACE_PSTATE_MACRO/aarch64_pstate_sm_body/ %s | opt -passes=loop-vectorize,slp-vectorizer -S - | FileCheck %s --check-prefix=CHECK
; RUN: sed -e s/REPLACE_PSTATE_MACRO/aarch64_pstate_sm_body/ %s | opt -passes=loop-vectorize,slp-vectorizer -S -enable-scalable-autovec-in-streaming-mode - | FileCheck %s --check-prefix=CHECK-FORCE-SCALABLE
; RUN: sed -e s/REPLACE_PSTATE_MACRO/aarch64_pstate_sm_body/ %s | opt -passes=loop-vectorize,slp-vectorizer -S -enable-fixedwidth-autovec-in-streaming-mode - | FileCheck %s --check-prefix=CHECK-FORCE-FIXEDWIDTH

target triple = "aarch64-unknown-linux-gnu"

attributes #0 = { vscale_range(1,16) "target-features"="+neon,+sme,+sve2" "REPLACE_PSTATE_MACRO" }

define void @test_fixedwidth_loopvec(ptr noalias %dst, ptr readonly %src, i32 %N) #0 {
; CHECK-LABEL: @test_fixedwidth_loopvec
; CHECK-NOT: <{{[1-9]+}} x i32>
; CHECK-FORCE-FIXEDWIDTH-LABEL: @test_fixedwidth_loopvec
; CHECK-FORCE-FIXEDWIDTH: <{{[1-9]+}} x i32>
entry:
  %cmp6 = icmp sgt i32 %N, 0
  br i1 %cmp6, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %N to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %src, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, 42
  %arrayidx2 = getelementptr inbounds i32, ptr %dst, i64 %indvars.iv
  store i32 %add, ptr %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !0
}

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.interleave.count", i32 1}
!3 = !{!"llvm.loop.vectorize.scalable.enable", i1 false}

define void @test_scalable_loopvec(ptr noalias %dst, ptr readonly %src, i32 %N) #0 {
; CHECK-LABEL: @test_scalable_loopvec
; CHECK-NOT: <vscale x {{[1-9]+}} x i32>
; CHECK-FORCE-SCALABLE-LABEL: @test_fixedwidth_loopvec
; CHECK-FORCE-SCALABLE-LABEL: <vscale x {{[1-9]+}} x i32>
entry:
  %cmp6 = icmp sgt i32 %N, 0
  br i1 %cmp6, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %N to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %src, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, 42
  %arrayidx2 = getelementptr inbounds i32, ptr %dst, i64 %indvars.iv
  store i32 %add, ptr %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !4
}

!4 = distinct !{!4, !5, !6, !7}
!5 = !{!"llvm.loop.mustprogress"}
!6 = !{!"llvm.loop.interleave.count", i32 1}
!7 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}

define void @test_slp(ptr noalias %dst, ptr readonly %src, i32 %N) #0 {
; CHECK-LABEL: @test_slp
; CHECK-NOT: <{{[1-9]+}} x i32>
; CHECK-FORCE-FIXEDWIDTH-LABEL: @test_slp
; CHECK-FORCE-FIXEDWIDTH: <{{[1-9]+}} x i32>
entry:
  %cmp6 = icmp sgt i32 %N, 0
  br i1 %cmp6, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %N to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %src, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, 42
  %arrayidx2 = getelementptr inbounds i32, ptr %dst, i64 %indvars.iv
  store i32 %add, ptr %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !8
}

!8 = distinct !{!8, !9, !10, !11}
!9 = !{!"llvm.loop.mustprogress"}
!10 = !{!"llvm.loop.interleave.count", i32 4}
!11 = !{!"llvm.loop.vectorize.width", i32 1}
