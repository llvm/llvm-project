; RUN: opt < %s -passes=loop-vectorize -sve-tail-folding=disabled -S | FileCheck %s -check-prefix=CHECK-NOTF
; RUN: opt < %s -passes=loop-vectorize -sve-tail-folding=default -S | FileCheck %s -check-prefix=CHECK-NOTF
; RUN: opt < %s -passes=loop-vectorize -sve-tail-folding=all -S | FileCheck %s -check-prefix=CHECK-TF
; RUN: opt < %s -passes=loop-vectorize -sve-tail-folding=disabled+simple+reductions+recurrences -S | FileCheck %s -check-prefix=CHECK-TF
; RUN: opt < %s -passes=loop-vectorize -sve-tail-folding=all+noreductions -S | FileCheck %s -check-prefix=CHECK-TF-NORED
; RUN: opt < %s -passes=loop-vectorize -sve-tail-folding=all+norecurrences -S | FileCheck %s -check-prefix=CHECK-TF-NOREC
; RUN: opt < %s -passes=loop-vectorize -sve-tail-folding=reductions -S | FileCheck %s -check-prefix=CHECK-TF-ONLYRED

target triple = "aarch64-unknown-linux-gnu"

define void @simple_memset(i32 %val, i32* %ptr, i64 %n) #0 {
; CHECK-NOTF-LABEL: @simple_memset(
; CHECK-NOTF:       vector.ph:
; CHECK-NOTF:         %[[INSERT:.*]] = insertelement <vscale x 4 x i32> poison, i32 %val, i32 0
; CHECK-NOTF:         %[[SPLAT:.*]] = shufflevector <vscale x 4 x i32> %[[INSERT]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NOTF:       vector.body:
; CHECK-NOTF-NOT:     %{{.*}} = phi <vscale x 4 x i1>
; CHECK-NOTF:         store <vscale x 4 x i32> %[[SPLAT]], <vscale x 4 x i32>*

; CHECK-TF-NORED-LABEL: @simple_memset(
; CHECK-TF-NORED:       vector.ph:
; CHECK-TF-NORED:         %[[INSERT:.*]] = insertelement <vscale x 4 x i32> poison, i32 %val, i32 0
; CHECK-TF-NORED:         %[[SPLAT:.*]] = shufflevector <vscale x 4 x i32> %[[INSERT]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-TF-NORED:       vector.body:
; CHECK-TF-NORED:         %[[ACTIVE_LANE_MASK:.*]] = phi <vscale x 4 x i1>
; CHECK-TF-NORED:         call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> %[[SPLAT]], {{.*}} %[[ACTIVE_LANE_MASK]]

; CHECK-TF-NOREC-LABEL: @simple_memset(
; CHECK-TF-NOREC:       vector.ph:
; CHECK-TF-NOREC:         %[[INSERT:.*]] = insertelement <vscale x 4 x i32> poison, i32 %val, i32 0
; CHECK-TF-NOREC:         %[[SPLAT:.*]] = shufflevector <vscale x 4 x i32> %[[INSERT]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-TF-NOREC:       vector.body:
; CHECK-TF-NOREC:         %[[ACTIVE_LANE_MASK:.*]] = phi <vscale x 4 x i1>
; CHECK-TF-NOREC:         call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> %[[SPLAT]], {{.*}} %[[ACTIVE_LANE_MASK]]

; CHECK-TF-LABEL: @simple_memset(
; CHECK-TF:       vector.ph:
; CHECK-TF:         %[[INSERT:.*]] = insertelement <vscale x 4 x i32> poison, i32 %val, i32 0
; CHECK-TF:         %[[SPLAT:.*]] = shufflevector <vscale x 4 x i32> %[[INSERT]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-TF:       vector.body:
; CHECK-TF:         %[[ACTIVE_LANE_MASK:.*]] = phi <vscale x 4 x i1>
; CHECK-TF:         call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> %[[SPLAT]], {{.*}} %[[ACTIVE_LANE_MASK]]

; CHECK-TF-ONLYRED-LABEL: @simple_memset(
; CHECK-TF-ONLYRED:       vector.ph:
; CHECK-TF-ONLYRED:         %[[INSERT:.*]] = insertelement <vscale x 4 x i32> poison, i32 %val, i32 0
; CHECK-TF-ONLYRED:         %[[SPLAT:.*]] = shufflevector <vscale x 4 x i32> %[[INSERT]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-TF-ONLYRED:       vector.body:
; CHECK-TF-ONLYRED-NOT:     %{{.*}} = phi <vscale x 4 x i1>
; CHECK-TF-ONLYRED:         store <vscale x 4 x i32> %[[SPLAT]], <vscale x 4 x i32>*

entry:
  br label %while.body

while.body:                                       ; preds = %while.body, %entry
  %index = phi i64 [ %index.next, %while.body ], [ 0, %entry ]
  %gep = getelementptr i32, i32* %ptr, i64 %index
  store i32 %val, i32* %gep
  %index.next = add nsw i64 %index, 1
  %cmp10 = icmp ult i64 %index.next, %n
  br i1 %cmp10, label %while.body, label %while.end.loopexit, !llvm.loop !0

while.end.loopexit:                               ; preds = %while.body
  ret void
}

define float @fadd_red_fast(float* noalias nocapture readonly %a, i64 %n) #0 {
; CHECK-NOTF-LABEL: @fadd_red_fast
; CHECK-NOTF:       vector.body:
; CHECK-NOTF-NOT:     %{{.*}} = phi <vscale x 4 x i1>
; CHECK-NOTF:         %[[LOAD:.*]] = load <vscale x 4 x float>
; CHECK-NOTF:         %[[ADD:.*]] = fadd fast <vscale x 4 x float> %[[LOAD]]
; CHECK-NOTF:       middle.block:
; CHECK-NOTF-NEXT:    call fast float @llvm.vector.reduce.fadd.nxv4f32(float -0.000000e+00, <vscale x 4 x float> %[[ADD]])

; CHECK-TF-NORED-LABEL: @fadd_red_fast
; CHECK-TF-NORED:       vector.body:
; CHECK-TF-NORED-NOT:     %{{.*}} = phi <vscale x 4 x i1>
; CHECK-TF-NORED:         %[[LOAD:.*]] = load <vscale x 4 x float>
; CHECK-TF-NORED:         %[[ADD:.*]] = fadd fast <vscale x 4 x float> %[[LOAD]]
; CHECK-TF-NORED:       middle.block:
; CHECK-TF-NORED-NEXT:    call fast float @llvm.vector.reduce.fadd.nxv4f32(float -0.000000e+00, <vscale x 4 x float> %[[ADD]])

; CHECK-TF-NOREC-LABEL: @fadd_red_fast
; CHECK-TF-NOREC:       vector.body:
; CHECK-TF-NOREC:         %[[ACTIVE_LANE_MASK:.*]] = phi <vscale x 4 x i1>
; CHECK-TF-NOREC:         %[[VEC_PHI:.*]] = phi <vscale x 4 x float>
; CHECK-TF-NOREC:         %[[LOAD:.*]] = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0nxv4f32(<vscale x 4 x float>{{.*}} %[[ACTIVE_LANE_MASK]]
; CHECK-TF-NOREC:         %[[ADD:.*]] = fadd fast <vscale x 4 x float> %[[LOAD]]
; CHECK-TF-NOREC:         %[[SEL:.*]] = select fast <vscale x 4 x i1> %[[ACTIVE_LANE_MASK]], <vscale x 4 x float> %[[ADD]], <vscale x 4 x float> %[[VEC_PHI]]
; CHECK-TF-NOREC:       middle.block:
; CHECK-TF-NOREC-NEXT:    call fast float @llvm.vector.reduce.fadd.nxv4f32(float -0.000000e+00, <vscale x 4 x float> %[[SEL]])

; CHECK-TF-LABEL: @fadd_red_fast
; CHECK-TF:       vector.body:
; CHECK-TF:         %[[ACTIVE_LANE_MASK:.*]] = phi <vscale x 4 x i1>
; CHECK-TF:         %[[VEC_PHI:.*]] = phi <vscale x 4 x float>
; CHECK-TF:         %[[LOAD:.*]] = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0nxv4f32(<vscale x 4 x float>{{.*}} %[[ACTIVE_LANE_MASK]]
; CHECK-TF:         %[[ADD:.*]] = fadd fast <vscale x 4 x float> %[[LOAD]]
; CHECK-TF:         %[[SEL:.*]] = select fast <vscale x 4 x i1> %[[ACTIVE_LANE_MASK]], <vscale x 4 x float> %[[ADD]], <vscale x 4 x float> %[[VEC_PHI]]
; CHECK-TF:       middle.block:
; CHECK-TF-NEXT:    call fast float @llvm.vector.reduce.fadd.nxv4f32(float -0.000000e+00, <vscale x 4 x float> %[[SEL]])

; CHECK-TF-ONLYRED-LABEL: @fadd_red_fast
; CHECK-TF-ONLYRED:       vector.body:
; CHECK-TF-ONLYRED:         %[[ACTIVE_LANE_MASK:.*]] = phi <vscale x 4 x i1>
; CHECK-TF-ONLYRED:         %[[VEC_PHI:.*]] = phi <vscale x 4 x float>
; CHECK-TF-ONLYRED:         %[[LOAD:.*]] = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0nxv4f32(<vscale x 4 x float>{{.*}} %[[ACTIVE_LANE_MASK]]
; CHECK-TF-ONLYRED:         %[[ADD:.*]] = fadd fast <vscale x 4 x float> %[[LOAD]]
; CHECK-TF-ONLYRED:         %[[SEL:.*]] = select fast <vscale x 4 x i1> %[[ACTIVE_LANE_MASK]], <vscale x 4 x float> %[[ADD]], <vscale x 4 x float> %[[VEC_PHI]]
; CHECK-TF-ONLYRED:       middle.block:
; CHECK-TF-ONLYRED-NEXT:    call fast float @llvm.vector.reduce.fadd.nxv4f32(float -0.000000e+00, <vscale x 4 x float> %[[SEL]])
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi float [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %a, i64 %iv
  %0 = load float, float* %arrayidx, align 4
  %add = fadd fast float %0, %sum.07
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret float %add
}

define void @add_recur(i32* noalias %dst, i32* noalias %src, i64 %n) #0 {
; CHECK-NOTF-LABEL: @add_recur
; CHECK-NOTF:       entry:
; CHECK-NOTF:         %[[PRE:.*]] = load i32, i32* %src, align 4
; CHECK-NOTF:       vector.ph:
; CHECK-NOTF:         %[[RECUR_INIT:.*]] = insertelement <vscale x 4 x i32> poison, i32 %[[PRE]]
; CHECK-NOTF:       vector.body:
; CHECK-NOTF-NOT:     %{{.*}} = phi <vscale x 4 x i1>
; CHECK-NOTF:         %[[VECTOR_RECUR:.*]] = phi <vscale x 4 x i32> [ %[[RECUR_INIT]], %vector.ph ], [ %[[LOAD:.*]], %vector.body ]
; CHECK-NOTF:         %[[LOAD]] = load <vscale x 4 x i32>
; CHECK-NOTF:         %[[SPLICE:.*]] = call <vscale x 4 x i32> @llvm.experimental.vector.splice.nxv4i32(<vscale x 4 x i32> %[[VECTOR_RECUR]], <vscale x 4 x i32> %[[LOAD]], i32 -1)
; CHECK-NOTF:         %[[ADD:.*]] = add nsw <vscale x 4 x i32> %[[LOAD]], %[[SPLICE]]
; CHECK-NOTF:         store <vscale x 4 x i32> %[[ADD]]

; CHECK-TF-NORED-LABEL: @add_recur
; CHECK-TF-NORED:       entry:
; CHECK-TF-NORED:         %[[PRE:.*]] = load i32, i32* %src, align 4
; CHECK-TF-NORED:       vector.ph:
; CHECK-TF-NORED:         %[[RECUR_INIT:.*]] = insertelement <vscale x 4 x i32> poison, i32 %[[PRE]]
; CHECK-TF-NORED:       vector.body:
; CHECK-TF-NORED:         %[[ACTIVE_LANE_MASK:.*]] = phi <vscale x 4 x i1>
; CHECK-TF-NORED:         %[[VECTOR_RECUR:.*]] = phi <vscale x 4 x i32> [ %[[RECUR_INIT]], %vector.ph ], [ %[[LOAD:.*]], %vector.body ]
; CHECK-TF-NORED:         %[[LOAD]] = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0nxv4i32(<vscale x 4 x i32>{{.*}} %[[ACTIVE_LANE_MASK]]
; CHECK-TF-NORED:         %[[SPLICE:.*]] = call <vscale x 4 x i32> @llvm.experimental.vector.splice.nxv4i32(<vscale x 4 x i32> %[[VECTOR_RECUR]], <vscale x 4 x i32> %[[LOAD]], i32 -1)
; CHECK-TF-NORED:         %[[ADD:.*]] = add nsw <vscale x 4 x i32> %[[LOAD]], %[[SPLICE]]
; CHECK-TF-NORED:         call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> %[[ADD]], {{.*}} <vscale x 4 x i1> %[[ACTIVE_LANE_MASK]])

; CHECK-TF-NOREC-LABEL: @add_recur
; CHECK-TF-NOREC:       entry:
; CHECK-TF-NOREC:         %[[PRE:.*]] = load i32, i32* %src, align 4
; CHECK-TF-NOREC:       vector.ph:
; CHECK-TF-NOREC:         %[[RECUR_INIT:.*]] = insertelement <vscale x 4 x i32> poison, i32 %[[PRE]]
; CHECK-TF-NOREC:       vector.body:
; CHECK-TF-NOREC-NOT:     %{{.*}} = phi <vscale x 4 x i1>
; CHECK-TF-NOREC:         %[[VECTOR_RECUR:.*]] = phi <vscale x 4 x i32> [ %[[RECUR_INIT]], %vector.ph ], [ %[[LOAD:.*]], %vector.body ]
; CHECK-TF-NOREC:         %[[LOAD]] = load <vscale x 4 x i32>
; CHECK-TF-NOREC:         %[[SPLICE:.*]] = call <vscale x 4 x i32> @llvm.experimental.vector.splice.nxv4i32(<vscale x 4 x i32> %[[VECTOR_RECUR]], <vscale x 4 x i32> %[[LOAD]], i32 -1)
; CHECK-TF-NOREC:         %[[ADD:.*]] = add nsw <vscale x 4 x i32> %[[LOAD]], %[[SPLICE]]
; CHECK-TF-NOREC:         store <vscale x 4 x i32> %[[ADD]]

; CHECK-TF-LABEL: @add_recur
; CHECK-TF:       entry:
; CHECK-TF:         %[[PRE:.*]] = load i32, i32* %src, align 4
; CHECK-TF:       vector.ph:
; CHECK-TF:         %[[RECUR_INIT:.*]] = insertelement <vscale x 4 x i32> poison, i32 %[[PRE]]
; CHECK-TF:       vector.body:
; CHECK-TF:         %[[ACTIVE_LANE_MASK:.*]] = phi <vscale x 4 x i1>
; CHECK-TF:         %[[VECTOR_RECUR:.*]] = phi <vscale x 4 x i32> [ %[[RECUR_INIT]], %vector.ph ], [ %[[LOAD:.*]], %vector.body ]
; CHECK-TF:         %[[LOAD]] = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0nxv4i32(<vscale x 4 x i32>{{.*}} %[[ACTIVE_LANE_MASK]]
; CHECK-TF:         %[[SPLICE:.*]] = call <vscale x 4 x i32> @llvm.experimental.vector.splice.nxv4i32(<vscale x 4 x i32> %[[VECTOR_RECUR]], <vscale x 4 x i32> %[[LOAD]], i32 -1)
; CHECK-TF:         %[[ADD:.*]] = add nsw <vscale x 4 x i32> %[[LOAD]], %[[SPLICE]]
; CHECK-TF:         call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> %[[ADD]], {{.*}} <vscale x 4 x i1> %[[ACTIVE_LANE_MASK]])

; CHECK-TF-ONLYRED-LABEL: @add_recur
; CHECK-TF-ONLYRED:       entry:
; CHECK-TF-ONLYRED:         %[[PRE:.*]] = load i32, i32* %src, align 4
; CHECK-TF-ONLYRED:       vector.ph:
; CHECK-TF-ONLYRED:         %[[RECUR_INIT:.*]] = insertelement <vscale x 4 x i32> poison, i32 %[[PRE]]
; CHECK-TF-ONLYRED:       vector.body:
; CHECK-TF-ONLYRED-NOT:     %{{.*}} = phi <vscale x 4 x i1>
; CHECK-TF-ONLYRED:         %[[VECTOR_RECUR:.*]] = phi <vscale x 4 x i32> [ %[[RECUR_INIT]], %vector.ph ], [ %[[LOAD:.*]], %vector.body ]
; CHECK-TF-ONLYRED:         %[[LOAD]] = load <vscale x 4 x i32>
; CHECK-TF-ONLYRED:         %[[SPLICE:.*]] = call <vscale x 4 x i32> @llvm.experimental.vector.splice.nxv4i32(<vscale x 4 x i32> %[[VECTOR_RECUR]], <vscale x 4 x i32> %[[LOAD]], i32 -1)
; CHECK-TF-ONLYRED:         %[[ADD:.*]] = add nsw <vscale x 4 x i32> %[[LOAD]], %[[SPLICE]]
; CHECK-TF-ONLYRED:         store <vscale x 4 x i32> %[[ADD]]

entry:
  %.pre = load i32, i32* %src, align 4
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %0 = phi i32 [ %1, %for.body ], [ %.pre, %entry ]
  %i.010 = phi i64 [ %add, %for.body ], [ 0, %entry ]
  %add = add nuw nsw i64 %i.010, 1
  %arrayidx1 = getelementptr inbounds i32, i32* %src, i64 %add
  %1 = load i32, i32* %arrayidx1, align 4
  %add2 = add nsw i32 %1, %0
  %arrayidx3 = getelementptr inbounds i32, i32* %dst, i64 %i.010
  store i32 %add2, i32* %arrayidx3, align 4
  %exitcond.not = icmp eq i64 %add, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:                                          ; preds = %for.body
  ret void
}

define void @interleave(float* noalias %dst, float* noalias %src, i64 %n) #0 {
; CHECK-NOTF-LABEL: @interleave(
; CHECK-NOTF:       vector.body:
; CHECK-NOTF:         %[[LOAD:.*]] = load <8 x float>, <8 x float>
; CHECK-NOTF:         %{{.*}} = shufflevector <8 x float> %[[LOAD]], <8 x float> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK-NOTF:         %{{.*}} = shufflevector <8 x float> %[[LOAD]], <8 x float> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>

; CHECK-TF-LABEL: @interleave(
; CHECK-TF:       vector.body:
; CHECK-TF:         %[[LOAD:.*]] = load <8 x float>, <8 x float>
; CHECK-TF:         %{{.*}} = shufflevector <8 x float> %[[LOAD]], <8 x float> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK-TF:         %{{.*}} = shufflevector <8 x float> %[[LOAD]], <8 x float> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>

; CHECK-TF-NORED-LABEL: @interleave(
; CHECK-TF-NORED:       vector.body:
; CHECK-TF-NORED:         %[[LOAD:.*]] = load <8 x float>, <8 x float>
; CHECK-TF-NORED:         %{{.*}} = shufflevector <8 x float> %[[LOAD]], <8 x float> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK-TF-NORED:         %{{.*}} = shufflevector <8 x float> %[[LOAD]], <8 x float> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>

; CHECK-TF-NOREC-LABEL: @interleave(
; CHECK-TF-NOREC:       vector.body:
; CHECK-TF-NOREC:         %[[LOAD:.*]] = load <8 x float>, <8 x float>
; CHECK-TF-NOREC:         %{{.*}} = shufflevector <8 x float> %[[LOAD]], <8 x float> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK-TF-NOREC:         %{{.*}} = shufflevector <8 x float> %[[LOAD]], <8 x float> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>

entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.021 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %mul = shl nuw nsw i64 %i.021, 1
  %arrayidx = getelementptr inbounds float, float* %src, i64 %mul
  %0 = load float, float* %arrayidx, align 4
  %mul1 = mul nuw nsw i64 %i.021, 3
  %arrayidx2 = getelementptr inbounds float, float* %dst, i64 %mul1
  store float %0, float* %arrayidx2, align 4
  %add = or i64 %mul, 1
  %arrayidx4 = getelementptr inbounds float, float* %src, i64 %add
  %1 = load float, float* %arrayidx4, align 4
  %add6 = add nuw nsw i64 %mul1, 1
  %arrayidx7 = getelementptr inbounds float, float* %dst, i64 %add6
  store float %1, float* %arrayidx7, align 4
  %add9 = add nuw nsw i64 %mul1, 2
  %arrayidx10 = getelementptr inbounds float, float* %dst, i64 %add9
  store float 3.000000e+00, float* %arrayidx10, align 4
  %inc = add nuw nsw i64 %i.021, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

attributes #0 = { "target-features"="+sve" }

!0 = distinct !{!0, !1, !2, !3, !4}
!1 = !{!"llvm.loop.vectorize.width", i32 4}
!2 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
!3 = !{!"llvm.loop.interleave.count", i32 1}
!4 = !{!"llvm.loop.vectorize.enable", i1 true}
