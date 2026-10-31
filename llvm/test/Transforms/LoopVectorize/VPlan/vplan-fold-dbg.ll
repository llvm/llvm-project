; RUN: opt -p debugify,loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -S \
; RUN:   -vplan-print-after=printOptimizedVPlan < %s 2> %t | FileCheck %s
; RUN: cat %t | FileCheck %s --check-prefix=VPLAN

; The debug attached to the urem instruction should remain after
; being folded to `and i64 %iv, 15`
define void @urem_fold(ptr %dst, ptr %src, i64 %n) {
; CHECK-LABEL: define void @urem_fold(
; CHECK:  [[VECTOR_BODY:.*:]]
; CHECK:    [[FOLD:%.*]] = and i64 {{.*}}, 15, !dbg [[DBG:![0-9]+]]
; CHECK:  [[LOOP:.*:]]
; CHECK:    [[ORIG:%.*]] = urem i64 [[IV:%.*]], 16, !dbg [[DBG]]
;
; VPLAN-LABEL: VPlan for loop in 'urem_fold' after printOptimizedVPlan
; VPLAN: EMIT vp{{.*}} = and vp{{.*}}, ir<15>, !dbg
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %urem = urem i64 %iv, 16
  %arrayidx = getelementptr inbounds nuw [4 x i8], ptr %src, i64 %urem
  %load1 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds nuw [4 x i8], ptr %dst, i64 %iv
  %load2 = load i32, ptr %arrayidx2, align 4
  %add = add nsw i32 %load2, %load1
  store i32 %add, ptr %arrayidx2, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void

}

define void @test_select_chain_debugloc(ptr noalias %dst, ptr noalias %a, ptr noalias %b, i64 %n) {
; CHECK-LABEL: define void @test_select_chain_debugloc(
; CHECK:  [[VECTOR_BODY:.*:]]
; CHECK:    [[VEC_SEL1:%.*]] = select <4 x i1> {{.*}}, <4 x i1> {{.*}}, <4 x i1> zeroinitializer
; CHECK:    [[VEC_SEL2:%.*]] = select <4 x i1> [[VEC_SEL1]], <4 x i32> {{.*}}, <4 x i32> {{.*}}, !dbg [[DBG_OUTER:![0-9]+]]
; CHECK:  [[LOOP:.*:]]
; CHECK:    [[ORIG_SEL_INNER:%.*]] = select i1 {{.*}}, i32 {{.*}}, i32 {{.*}}, !dbg [[DBG_INNER:![0-9]+]]
; CHECK:    [[ORIG_SEL_OUTER:%.*]] = select i1 {{.*}}, i32 [[ORIG_SEL_INNER]], i32 %lb, !dbg [[DBG_OUTER]]
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  %la = load i32, ptr %gep.a, align 4
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv
  %lb = load i32, ptr %gep.b, align 4
  %m0 = icmp sgt i32 %la, 0
  %m1 = icmp slt i32 %lb, 100
  %inner = select i1 %m1, i32 %la, i32 %lb
  %outer = select i1 %m0, i32 %inner, i32 %lb
  %gep.dst = getelementptr inbounds i32, ptr %dst, i64 %iv
  store i32 %outer, ptr %gep.dst, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}
