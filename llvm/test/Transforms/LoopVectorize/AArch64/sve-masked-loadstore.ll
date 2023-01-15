; RUN: opt -passes=loop-vectorize,dce,instcombine -mtriple aarch64-linux-gnu -mattr=+sve \
; RUN:   -prefer-predicate-over-epilogue=scalar-epilogue -S %s -o - | FileCheck %s

define void @mloadstore_f32(ptr noalias nocapture %a, ptr noalias nocapture readonly %b, i64 %n) {
; CHECK-LABEL: @mloadstore_f32
; CHECK: vector.body:
; CHECK:       %[[LOAD1:.*]] = load <vscale x 4 x float>, ptr
; CHECK-NEXT:  %[[MASK:.*]] = fcmp ogt <vscale x 4 x float> %[[LOAD1]],
; CHECK-NEXT:  %[[GEPA:.*]] = getelementptr float, ptr %a,
; CHECK-NEXT:  %[[LOAD2:.*]] = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0(ptr %[[GEPA]], i32 4, <vscale x 4 x i1> %[[MASK]]
; CHECK-NEXT:  %[[FADD:.*]] = fadd <vscale x 4 x float> %[[LOAD1]], %[[LOAD2]]
; CHECK-NEXT:  call void @llvm.masked.store.nxv4f32.p0(<vscale x 4 x float> %[[FADD]], ptr %[[GEPA]], i32 4, <vscale x 4 x i1> %[[MASK]])
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %i.011 = phi i64 [ %inc, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, ptr %b, i64 %i.011
  %0 = load float, ptr %arrayidx, align 4
  %cmp1 = fcmp ogt float %0, 0.000000e+00
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %arrayidx3 = getelementptr inbounds float, ptr %a, i64 %i.011
  %1 = load float, ptr %arrayidx3, align 4
  %add = fadd float %0, %1
  store float %add, ptr %arrayidx3, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %inc = add nuw nsw i64 %i.011, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body, !llvm.loop !0

exit:                                 ; preds = %for.inc
  ret void
}

define void @mloadstore_i32(ptr noalias nocapture %a, ptr noalias nocapture readonly %b, i64 %n) {
; CHECK-LABEL: @mloadstore_i32
; CHECK: vector.body:
; CHECK:       %[[LOAD1:.*]] = load <vscale x 4 x i32>, ptr
; CHECK-NEXT:  %[[MASK:.*]] = icmp ne <vscale x 4 x i32> %[[LOAD1]],
; CHECK-NEXT:  %[[GEPA:.*]] = getelementptr i32, ptr %a,
; CHECK-NEXT:  %[[LOAD2:.*]] = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0(ptr %[[GEPA]], i32 4, <vscale x 4 x i1> %[[MASK]]
; CHECK-NEXT:  %[[FADD:.*]] = add <vscale x 4 x i32> %[[LOAD1]], %[[LOAD2]]
; CHECK-NEXT:  call void @llvm.masked.store.nxv4i32.p0(<vscale x 4 x i32> %[[FADD]], ptr %[[GEPA]], i32 4, <vscale x 4 x i1> %[[MASK]])
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %i.011 = phi i64 [ %inc, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %i.011
  %0 = load i32, ptr %arrayidx, align 4
  %cmp1 = icmp ne i32 %0, 0
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %arrayidx3 = getelementptr inbounds i32, ptr %a, i64 %i.011
  %1 = load i32, ptr %arrayidx3, align 4
  %add = add i32 %0, %1
  store i32 %add, ptr %arrayidx3, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %inc = add nuw nsw i64 %i.011, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body, !llvm.loop !0

exit:                                 ; preds = %for.inc
  ret void
}

!0 = distinct !{!0, !1, !2, !3, !4, !5}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.vectorize.width", i32 4}
!3 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
!4 = !{!"llvm.loop.interleave.count", i32 1}
!5 = !{!"llvm.loop.vectorize.enable", i1 true}
