; RUN: opt < %s -scalable-vectorization=on -force-target-supports-scalable-vectors=true -loop-vectorize -force-vector-width=2 -force-vector-interleave=2  -S | FileCheck %s

define void @test1(float* noalias nocapture %a, float* noalias nocapture readonly %b) {
; CHECK-LABEL: @test1(
; CHECK:       vector.body:
; CHECK:         [[FCMP1:%.*]] = fcmp ogt <vscale x 2 x float>
; CHECK-NEXT:    [[FCMP2:%.*]] = fcmp ogt <vscale x 2 x float>
; CHECK-NEXT:    [[FCMP1L0:%.*]] = extractelement <vscale x 2 x i1> [[FCMP1]], i32 0
; CHECK-NEXT:    tail call void @llvm.assume(i1 [[FCMP1L0]])
; CHECK-NEXT:    [[FCMP2L0:%.*]] = extractelement <vscale x 2 x i1> [[FCMP2]], i32 0
; CHECK-NEXT:    tail call void @llvm.assume(i1 [[FCMP2L0]])
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %b, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %cmp1 = fcmp ogt float %0, 1.000000e+02
  tail call void @llvm.assume(i1 %cmp1)
  %add = fadd float %0, 1.000000e+00
  %arrayidx5 = getelementptr inbounds float, float* %a, i64 %indvars.iv
  store float %add, float* %arrayidx5, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, 1599
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !0

for.end:                                          ; preds = %for.body
  ret void
}

declare void @llvm.assume(i1) #0

attributes #0 = { nounwind willreturn }

%struct.data = type { float*, float* }

define void @test2(float *%a, float *%b) {
; CHECK-LABEL: @test2(
; CHECK:       entry:
; CHECK:         [[MASKCOND:%.*]] = icmp eq i64 %ptrint1, 0
; CHECK:         [[MASKCOND4:%.*]] = icmp eq i64 %ptrint2, 0
; CHECK:       vector.body:
; CHECK:         tail call void @llvm.assume(i1 [[MASKCOND]])
; CHECK-NEXT:    tail call void @llvm.assume(i1 [[MASKCOND]])
; CHECK:         tail call void @llvm.assume(i1 [[MASKCOND4]])
; CHECK-NEXT:    tail call void @llvm.assume(i1 [[MASKCOND4]])
entry:
  %ptrint1 = ptrtoint float* %a to i64
  %maskcond = icmp eq i64 %ptrint1, 0
  %ptrint2 = ptrtoint float* %b to i64
  %maskcond4 = icmp eq i64 %ptrint2, 0
  br label %for.body


for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  tail call void @llvm.assume(i1 %maskcond)
  %arrayidx = getelementptr inbounds float, float* %a, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %add = fadd float %0, 1.000000e+00
  tail call void @llvm.assume(i1 %maskcond4)
  %arrayidx5 = getelementptr inbounds float, float* %b, i64 %indvars.iv
  store float %add, float* %arrayidx5, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, 1599
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !0

for.end:                                          ; preds = %for.body
  ret void
}

; Test case for PR43620. Make sure we can vectorize with predication in presence
; of assume calls. For now, check that we drop all assumes in predicated blocks
; in the vector body.
define void @predicated_assume(float* noalias nocapture readonly %a, float* noalias nocapture %b, i64 %n) {
; Check that the vector.body does not contain any assumes.
; CHECK-LABEL: @predicated_assume(
; CHECK:       vector.body:
; CHECK-NOT:     llvm.assume
; CHECK:       for.body:
entry:
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %if.end5
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %if.end5 ]
  %cmp1 = icmp ult i64 %indvars.iv, 495616
  br i1 %cmp1, label %if.end5, label %if.else

if.else:                                          ; preds = %for.body
  %cmp2 = icmp ult i64 %indvars.iv, 991232
  tail call void @llvm.assume(i1 %cmp2)
  br label %if.end5

if.end5:                                          ; preds = %for.body, %if.else
  %x.0 = phi float [ 4.200000e+01, %if.else ], [ 2.300000e+01, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %a, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %mul = fmul float %x.0, %0
  %arrayidx7 = getelementptr inbounds float, float* %b, i64 %indvars.iv
  store float %mul, float* %arrayidx7, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cmp = icmp eq i64 %indvars.iv.next, %n
  br i1 %cmp, label %for.cond.cleanup, label %for.body, !llvm.loop !0

for.cond.cleanup:                                 ; preds = %if.end5, %entry
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
