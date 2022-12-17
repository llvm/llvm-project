; RUN: opt < %s -scalable-vectorization=on -force-target-supports-scalable-vectors=true -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=2  -S | FileCheck %s

define void @test1(float* noalias nocapture %a, float* noalias nocapture readonly %b) {
entry:
  br label %for.body

; CHECK-LABEL: @test1
; CHECK: vector.body:
; CHECK: @llvm.experimental.noalias.scope.decl
; CHECK-NOT: @llvm.experimental.noalias.scope.decl
; CHECK: for.body:
; CHECK: @llvm.experimental.noalias.scope.decl
; CHECK-NOT: @llvm.experimental.noalias.scope.decl
; CHECK: ret void

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %b, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %cmp1 = fcmp ogt float %0, 1.000000e+02
  tail call void @llvm.experimental.noalias.scope.decl(metadata !0)
  %add = fadd float %0, 1.000000e+00
  %arrayidx5 = getelementptr inbounds float, float* %a, i64 %indvars.iv
  store float %add, float* %arrayidx5, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, 1599
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !5

for.end:                                          ; preds = %for.body
  ret void
}

declare void @llvm.experimental.noalias.scope.decl(metadata)

%struct.data = type { float*, float* }

define void @test2(float* %a, float* %b) {
; CHECK-LABEL: @test2
; CHECK: vector.body:
; CHECK: @llvm.experimental.noalias.scope.decl(metadata [[SCOPE0_LIST:!.*]])
; CHECK: @llvm.experimental.noalias.scope.decl(metadata [[SCOPE4_LIST:!.*]])
; CHECK-NOT: @llvm.experimental.noalias.scope.decl
; CHECK: for.body:
; CHECK: @llvm.experimental.noalias.scope.decl(metadata [[SCOPE0_LIST]])
; CHECK: @llvm.experimental.noalias.scope.decl(metadata [[SCOPE4_LIST]])
; CHECK-NOT: @llvm.experimental.noalias.scope.decl
; CHECK: ret void
entry:
  %ptrint = ptrtoint float* %b to i64
  %maskcond = icmp eq i64 %ptrint, 0
  %ptrint2 = ptrtoint float* %a to i64
  %maskcond4 = icmp eq i64 %ptrint2, 0
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  tail call void @llvm.experimental.noalias.scope.decl(metadata !0)
  %arrayidx = getelementptr inbounds float, float* %b, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %add = fadd float %0, 1.000000e+00
  tail call void @llvm.experimental.noalias.scope.decl(metadata !4)
  %arrayidx5 = getelementptr inbounds float, float* %a, i64 %indvars.iv
  store float %add, float* %arrayidx5, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, 1599
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !5

for.end:                                          ; preds = %for.body
  ret void
}

define void @predicated_noalias_scope_decl(float* noalias nocapture readonly %a, float* noalias nocapture %b, i64 %n) {

; Check that the vector.body still contains a llvm.experimental.noalias.scope.decl

; CHECK-LABEL: @predicated_noalias_scope_decl(
; CHECK:   vector.body:
; CHECK:   call void @llvm.experimental.noalias.scope.decl
; CHECK-NOT: @llvm.experimental.noalias.scope.decl
; CHECK:   scalar.ph:
; CHECK-NOT: @llvm.experimental.noalias.scope.decl
; CHECK:   if.else:
; CHECK:   call void @llvm.experimental.noalias.scope.decl
; CHECK-NOT: @llvm.experimental.noalias.scope.decl
; CHECK: }

entry:
  br label %for.body

for.body:                                         ; preds = %entry, %if.end5
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %if.end5 ]
  %cmp1 = icmp ult i64 %indvars.iv, 495616
  br i1 %cmp1, label %if.end5, label %if.else

if.else:                                          ; preds = %for.body
  %cmp2 = icmp ult i64 %indvars.iv, 991232
  tail call void @llvm.experimental.noalias.scope.decl(metadata !0)
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
  br i1 %cmp, label %for.cond.cleanup, label %for.body, !llvm.loop !5

for.cond.cleanup:                                 ; preds = %if.end5
  ret void
}

!0 = !{ !1 }
!1 = distinct !{ !1, !2 }
!2 = distinct !{ !2 }
!3 = distinct !{ !3, !2 }
!4 = !{ !3 }
!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}

; CHECK: [[SCOPE0_LIST]] = !{[[SCOPE0:!.*]]}
; CHECK: [[SCOPE0]] = distinct !{[[SCOPE0]], [[SCOPE0_DOM:!.*]]}
; CHECK: [[SCOPE0_DOM]] = distinct !{[[SCOPE0_DOM]]}
; CHECK: [[SCOPE4_LIST]] = !{[[SCOPE4:!.*]]}
; CHECK: [[SCOPE4]] = distinct !{[[SCOPE4]], [[SCOPE0_DOM]]}
