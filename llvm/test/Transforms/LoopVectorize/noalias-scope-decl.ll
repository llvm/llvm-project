; RUN: opt < %s -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=2  -S | FileCheck %s

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
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

declare void @llvm.experimental.noalias.scope.decl(metadata)

%struct.data = type { float*, float* }

define void @test2(%struct.data* nocapture readonly %d) {
entry:
  %b = getelementptr inbounds %struct.data, %struct.data* %d, i64 0, i32 1
  %0 = load float*, float** %b, align 8
  %ptrint = ptrtoint float* %0 to i64
  %maskedptr = and i64 %ptrint, 31
  %maskcond = icmp eq i64 %maskedptr, 0
  %a = getelementptr inbounds %struct.data, %struct.data* %d, i64 0, i32 0
  %1 = load float*, float** %a, align 8
  %ptrint2 = ptrtoint float* %1 to i64
  %maskedptr3 = and i64 %ptrint2, 31
  %maskcond4 = icmp eq i64 %maskedptr3, 0
  br label %for.body

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

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  tail call void @llvm.experimental.noalias.scope.decl(metadata !0)
  %arrayidx = getelementptr inbounds float, float* %0, i64 %indvars.iv
  %2 = load float, float* %arrayidx, align 4
  %add = fadd float %2, 1.000000e+00
  tail call void @llvm.experimental.noalias.scope.decl(metadata !4)
  %arrayidx5 = getelementptr inbounds float, float* %1, i64 %indvars.iv
  store float %add, float* %arrayidx5, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, 1599
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

define void @predicated_noalias_scope_decl(float* noalias nocapture readonly %a, float* noalias nocapture %b, i32 %n) {

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
  %cmp15 = icmp eq i32 %n, 0
  br i1 %cmp15, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %0 = zext i32 %n to i64
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %if.end5
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %if.end5
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %if.end5 ]
  %cmp1 = icmp ult i64 %indvars.iv, 495616
  br i1 %cmp1, label %if.end5, label %if.else

if.else:                                          ; preds = %for.body
  %cmp2 = icmp ult i64 %indvars.iv, 991232
  tail call void @llvm.experimental.noalias.scope.decl(metadata !0)
  br label %if.end5

if.end5:                                          ; preds = %for.body, %if.else
  %x.0 = phi float [ 4.200000e+01, %if.else ], [ 2.300000e+01, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %a, i64 %indvars.iv
  %1 = load float, float* %arrayidx, align 4
  %mul = fmul float %x.0, %1
  %arrayidx7 = getelementptr inbounds float, float* %b, i64 %indvars.iv
  store float %mul, float* %arrayidx7, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cmp = icmp eq i64 %indvars.iv.next, %0
  br i1 %cmp, label %for.cond.cleanup.loopexit, label %for.body
}

!0 = !{ !1 }
!1 = distinct !{ !1, !2 }
!2 = distinct !{ !2 }
!3 = distinct !{ !3, !2 }
!4 = !{ !3 }

; CHECK: [[SCOPE0_LIST]] = !{[[SCOPE0:!.*]]}
; CHECK: [[SCOPE0]] = distinct !{[[SCOPE0]], [[SCOPE0_DOM:!.*]]}
; CHECK: [[SCOPE0_DOM]] = distinct !{[[SCOPE0_DOM]]}
; CHECK: [[SCOPE4_LIST]] = !{[[SCOPE4:!.*]]}
; CHECK: [[SCOPE4]] = distinct !{[[SCOPE4]], [[SCOPE0_DOM]]}
