; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -S \
; RUN:   -prefer-predicate-over-epilogue=scalar-epilogue < %s | FileCheck %s --check-prefix=CHECK-VF4IC1
; RUN: opt -passes=loop-vectorize -force-vector-interleave=4 -force-vector-width=4 -S \
; RUN:   -prefer-predicate-over-epilogue=scalar-epilogue < %s | FileCheck %s --check-prefix=CHECK-VF4IC4

target triple = "aarch64-linux-gnu"

define i32 @select_const_i32_from_icmp(ptr nocapture readonly %v, i64 %n) #0 {
; CHECK-VF4IC1-LABEL: @select_const_i32_from_icmp
; CHECK-VF4IC1:      vector.body:
; CHECK-VF4IC1:        [[VEC_PHI:%.*]] = phi <vscale x 4 x i1> [ zeroinitializer, %vector.ph ], [ [[VEC_SEL:%.*]], %vector.body ]
; CHECK-VF4IC1:        [[VEC_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK-VF4IC1-NEXT:   [[VEC_ICMP:%.*]] = icmp eq <vscale x 4 x i32> [[VEC_LOAD]], splat (i32 3)
; CHECK-VF4IC1-NEXT:   [[NOT:%*]] = xor <vscale x 4 x i1> [[VEC_ICMP]], splat (i1 true)
; CHECK-VF4IC1-NEXT:   [[VEC_SEL]] = or <vscale x 4 x i1> [[VEC_PHI]], [[NOT]]
; CHECK-VF4IC1:      middle.block:
; CHECK-VF4IC1-NEXT:   [[OR_RDX:%.*]] = call i1 @llvm.vector.reduce.or.nxv4i1(<vscale x 4 x i1> [[VEC_SEL]])
; CHECK-VF4IC1-NEXT:   [[FR:%.*]] = freeze i1 [[OR_RDX]]
; CHECK-VF4IC1-NEXT:   {{.*}} = select i1 [[FR]], i32 7, i32 3
; CHECK-VF4IC1:        %cmp.n = icmp eq i64 %n, %n.vec

; CHECK-VF4IC4-LABEL: @select_const_i32_from_icmp
; CHECK-VF4IC4:      vector.body:
; CHECK-VF4IC4:        [[VEC_PHI1:%.*]] = phi <vscale x 4 x i1> [ zeroinitializer, %vector.ph ], [ [[VEC_SEL1:%.*]], %vector.body ]
; CHECK-VF4IC4:        [[VEC_PHI2:%.*]] = phi <vscale x 4 x i1> [ zeroinitializer, %vector.ph ], [ [[VEC_SEL2:%.*]], %vector.body ]
; CHECK-VF4IC4:        [[VEC_PHI3:%.*]] = phi <vscale x 4 x i1> [ zeroinitializer, %vector.ph ], [ [[VEC_SEL3:%.*]], %vector.body ]
; CHECK-VF4IC4:        [[VEC_PHI4:%.*]] = phi <vscale x 4 x i1> [ zeroinitializer, %vector.ph ], [ [[VEC_SEL4:%.*]], %vector.body ]
; CHECK-VF4IC4:        [[VEC_ICMP1:%.*]] = icmp eq <vscale x 4 x i32> {{.*}}, splat (i32 3)
; CHECK-VF4IC4-NEXT:   [[VEC_ICMP2:%.*]] = icmp eq <vscale x 4 x i32> {{.*}}, splat (i32 3)
; CHECK-VF4IC4-NEXT:   [[VEC_ICMP3:%.*]] = icmp eq <vscale x 4 x i32> {{.*}}, splat (i32 3)
; CHECK-VF4IC4-NEXT:   [[VEC_ICMP4:%.*]] = icmp eq <vscale x 4 x i32> {{.*}}, splat (i32 3)
; CHECK-VF4IC4-NEXT:    [[NOT1:%.*]] = xor <vscale x 4 x i1> [[VEC_ICMP1]], splat (i1 true)
; CHECK-VF4IC4-NEXT:    [[NOT2:%.*]] = xor <vscale x 4 x i1> [[VEC_ICMP2]], splat (i1 true)
; CHECK-VF4IC4-NEXT:    [[NOT3:%.*]] = xor <vscale x 4 x i1> [[VEC_ICMP3]], splat (i1 true)
; CHECK-VF4IC4-NEXT:    [[NOT4:%.*]] = xor <vscale x 4 x i1> [[VEC_ICMP4]], splat (i1 true)
; CHECK-VF4IC4-NEXT:   [[VEC_SEL1:%.*]] = or <vscale x 4 x i1> [[VEC_PHI1]], [[NOT1]]
; CHECK-VF4IC4-NEXT:   [[VEC_SEL2:%.*]] = or <vscale x 4 x i1> [[VEC_PHI2]], [[NOT2]]
; CHECK-VF4IC4-NEXT:   [[VEC_SEL3:%.*]] = or <vscale x 4 x i1> [[VEC_PHI3]], [[NOT3]]
; CHECK-VF4IC4-NEXT:   [[VEC_SEL4:%.*]] = or <vscale x 4 x i1> [[VEC_PHI4]], [[NOT4]]
; CHECK-VF4IC4:      middle.block:
; CHECK-VF4IC4-NEXT:   [[OR1:%.*]] = or <vscale x 4 x i1> [[VEC_SEL2]], [[VEC_SEL1]]
; CHECK-VF4IC4-NEXT:   [[OR2:%.*]] = or <vscale x 4 x i1> [[VEC_SEL3]], [[OR1]]
; CHECK-VF4IC4-NEXT:   [[OR3:%.*]] = or <vscale x 4 x i1> [[VEC_SEL4]], [[OR2]]
; CHECK-VF4IC4-NEXT:   [[OR_RDX:%.*]] = call i1 @llvm.vector.reduce.or.nxv4i1(<vscale x 4 x i1> [[OR3]])
; CHECK-VF4IC4-NEXT:   [[FR:%.*]] = freeze i1 [[OR_RDX]]
; CHECK-VF4IC4-NEXT:   {{.*}} = select i1 [[FR]], i32 7, i32 3
; CHECK-VF4IC4-NEXT:   %cmp.n = icmp eq i64 %n, %n.vec
entry:
  br label %for.body

for.body:                                      ; preds = %entry, %for.body
  %0 = phi i64 [ 0, %entry ], [ %6, %for.body ]
  %1 = phi i32 [ 3, %entry ], [ %5, %for.body ]
  %2 = getelementptr inbounds i32, ptr %v, i64 %0
  %3 = load i32, ptr %2, align 4
  %4 = icmp eq i32 %3, 3
  %5 = select i1 %4, i32 %1, i32 7
  %6 = add nuw nsw i64 %0, 1
  %7 = icmp eq i64 %6, %n
  br i1 %7, label %exit, label %for.body, !llvm.loop !0

exit:                                     ; preds = %for.body
  ret i32 %5
}

define i32 @select_i32_from_icmp(ptr nocapture readonly %v, i32 %a, i32 %b, i64 %n) #0 {
; CHECK-VF4IC1-LABEL: @select_i32_from_icmp
; CHECK-VF4IC1:      vector.ph:
; CHECK-VF4IC1-NOT:    shufflevector <vscale x 4 x i32>
; CHECK-VF4IC1-NOT:    shufflevector <vscale x 4 x i32>
; CHECK-VF4IC1:      vector.body:
; CHECK-VF4IC1:        [[VEC_PHI:%.*]] = phi <vscale x 4 x i1> [ zeroinitializer, %vector.ph ], [ [[VEC_SEL:%.*]], %vector.body ]
; CHECK-VF4IC1:        [[VEC_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK-VF4IC1-NEXT:   [[VEC_ICMP:%.*]] = icmp eq <vscale x 4 x i32> [[VEC_LOAD]], splat (i32 3)
; CHECK-VF4IC1-NEXT:   [[NOT:%*]] = xor <vscale x 4 x i1> [[VEC_ICMP]], splat (i1 true)
; CHECK-VF4IC1-NEXT:   [[VEC_SEL]] = or <vscale x 4 x i1> [[VEC_PHI]], [[NOT]]
; CHECK-VF4IC1:      middle.block:
; CHECK-VF4IC1-NEXT:   [[OR_RDX:%.*]] = call i1 @llvm.vector.reduce.or.nxv4i1(<vscale x 4 x i1> [[VEC_SEL]])
; CHECK-VF4IC1-NEXT:   [[FR:%.*]] = freeze i1 [[OR_RDX]]
; CHECK-VF4IC1-NEXT:   {{.*}} = select i1 [[FR]], i32 %b, i32 %a
; CHECK-VF4IC1:        %cmp.n = icmp eq i64 %n, %n.vec

; CHECK-VF4IC4-LABEL: @select_i32_from_icmp
; CHECK-VF4IC4:      vector.body:
entry:
  br label %for.body

for.body:                                      ; preds = %entry, %for.body
  %0 = phi i64 [ 0, %entry ], [ %6, %for.body ]
  %1 = phi i32 [ %a, %entry ], [ %5, %for.body ]
  %2 = getelementptr inbounds i32, ptr %v, i64 %0
  %3 = load i32, ptr %2, align 4
  %4 = icmp eq i32 %3, 3
  %5 = select i1 %4, i32 %1, i32 %b
  %6 = add nuw nsw i64 %0, 1
  %7 = icmp eq i64 %6, %n
  br i1 %7, label %exit, label %for.body, !llvm.loop !0

exit:                                     ; preds = %for.body
  ret i32 %5
}

define i32 @select_const_i32_from_fcmp(ptr nocapture readonly %v, i64 %n) #0 {
; CHECK-VF4IC1-LABEL: @select_const_i32_from_fcmp
; CHECK-VF4IC1:      vector.body:
; CHECK-VF4IC1:        [[VEC_PHI:%.*]] = phi <vscale x 4 x i1> [ zeroinitializer, %vector.ph ], [ [[VEC_SEL:%.*]], %vector.body ]
; CHECK-VF4IC1:        [[VEC_LOAD:%.*]] = load <vscale x 4 x float>
; CHECK-VF4IC1-NEXT:   [[VEC_ICMP:%.*]] = fcmp fast ueq <vscale x 4 x float> [[VEC_LOAD]], splat (float 3.000000e+00)
; CHECK-VF4IC1-NEXT:   [[NOT:%*]] = xor <vscale x 4 x i1> [[VEC_ICMP]], splat (i1 true)
; CHECK-VF4IC1-NEXT:   [[VEC_SEL]] = or <vscale x 4 x i1> [[VEC_PHI]], [[NOT]]
; CHECK-VF4IC1:      middle.block:
; CHECK-VF4IC1-NEXT:   [[OR_RDX:%.*]] = call i1 @llvm.vector.reduce.or.nxv4i1(<vscale x 4 x i1> [[VEC_SEL]])
; CHECK-VF4IC1-NEXT:   [[FR:%.*]] = freeze i1 [[OR_RDX]]
; CHECK-VF4IC1-NEXT:   {{.*}} = select i1 [[FR]], i32 1, i32 2
; CHECK-VF4IC1:        %cmp.n = icmp eq i64 %n, %n.vec

; CHECK-VF4IC4-LABEL: @select_const_i32_from_fcmp
; CHECK-VF4IC4:      vector.body:
entry:
  br label %for.body

for.body:                                      ; preds = %entry, %for.body
  %0 = phi i64 [ 0, %entry ], [ %6, %for.body ]
  %1 = phi i32 [ 2, %entry ], [ %5, %for.body ]
  %2 = getelementptr inbounds float, ptr %v, i64 %0
  %3 = load float, ptr %2, align 4
  %4 = fcmp fast ueq float %3, 3.0
  %5 = select i1 %4, i32 %1, i32 1
  %6 = add nuw nsw i64 %0, 1
  %7 = icmp eq i64 %6, %n
  br i1 %7, label %exit, label %for.body, !llvm.loop !0

exit:                                     ; preds = %for.body
  ret i32 %5
}

define float @select_const_f32_from_icmp(ptr nocapture readonly %v, i64 %n) #0 {
; CHECK-VF4IC1-LABEL: @select_const_f32_from_icmp
; CHECK-VF4IC1-NOT: vector.body
; CHECK-VF4IC4-LABEL: @select_const_f32_from_icmp
; CHECK-VF4IC4-NOT: vector.body
entry:
  br label %for.body

for.body:                                      ; preds = %entry, %for.body
  %0 = phi i64 [ 0, %entry ], [ %6, %for.body ]
  %1 = phi fast float [ 3.0, %entry ], [ %5, %for.body ]
  %2 = getelementptr inbounds i32, ptr %v, i64 %0
  %3 = load i32, ptr %2, align 4
  %4 = icmp eq i32 %3, 3
  %5 = select fast i1 %4, float %1, float 7.0
  %6 = add nuw nsw i64 %0, 1
  %7 = icmp eq i64 %6, %n
  br i1 %7, label %exit, label %for.body, !llvm.loop !0

exit:                                     ; preds = %for.body
  ret float %5
}

define i32 @pred_select_const_i32_from_icmp(ptr noalias nocapture readonly %src1, ptr noalias nocapture readonly %src2, i64 %n) #0 {
; CHECK-VF4IC1-LABEL: @pred_select_const_i32_from_icmp
; CHECK-VF4IC1:      vector.body:
; CHECK-VF4IC1:        [[VEC_PHI:%.*]] = phi <vscale x 4 x i1> [ zeroinitializer, %vector.ph ], [ [[VEC_SEL:%.*]], %vector.body ]
; CHECK-VF4IC1:        [[VEC_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK-VF4IC1:        [[MASK:%.*]] = icmp sgt <vscale x 4 x i32> [[VEC_LOAD]], splat (i32 35)
; CHECK-VF4IC1:        [[MASKED_LOAD:%.*]] = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0(ptr {{%.*}}, i32 4, <vscale x 4 x i1> [[MASK]], <vscale x 4 x i32> poison)
; CHECK-VF4IC1-NEXT:   [[VEC_ICMP:%.*]] = icmp eq <vscale x 4 x i32> [[MASKED_LOAD]], splat (i32 2)
; CHECK-VF4IC1-NEXT:   [[VEC_SEL_TMP:%.*]] = or <vscale x 4 x i1> [[VEC_PHI]], [[VEC_ICMP]]
; CHECK-VF4IC1:        [[VEC_SEL:%.*]] = select <vscale x 4 x i1> [[MASK]], <vscale x 4 x i1> [[VEC_SEL_TMP]], <vscale x 4 x i1> [[VEC_PHI]]
; CHECK-VF4IC1:      middle.block:
; CHECK-VF4IC1-NEXT:   [[OR_RDX:%.*]] = call i1 @llvm.vector.reduce.or.nxv4i1(<vscale x 4 x i1> [[VEC_SEL]])
; CHECK-VF4IC1-NEXT:   [[FR:%.*]] = freeze i1 [[OR_RDX]]
; CHECK-VF4IC1-NEXT:   {{.*}} = select i1 [[FR]], i32 1, i32 0
; CHECK-VF4IC1:        %cmp.n = icmp eq i64 %n, %n.vec

; CHECK-VF4IC4-LABEL: @pred_select_const_i32_from_icmp
; CHECK-VF4IC4:      vector.body:
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %i.013 = phi i64 [ %inc, %for.inc ], [ 0, %entry ]
  %r.012 = phi i32 [ %r.1, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %src1, i64 %i.013
  %0 = load i32, ptr %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 35
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %arrayidx2 = getelementptr inbounds i32, ptr %src2, i64 %i.013
  %1 = load i32, ptr %arrayidx2, align 4
  %cmp3 = icmp eq i32 %1, 2
  %spec.select = select i1 %cmp3, i32 1, i32 %r.012
  br label %for.inc

for.inc:                                          ; preds = %if.then, %for.body
  %r.1 = phi i32 [ %r.012, %for.body ], [ %spec.select, %if.then ]
  %inc = add nuw nsw i64 %i.013, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %for.end.loopexit, label %for.body, !llvm.loop !0

for.end.loopexit:                                 ; preds = %for.inc
  %r.1.lcssa = phi i32 [ %r.1, %for.inc ]
  ret i32 %r.1.lcssa
}


attributes #0 = { "target-features"="+sve" }

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
