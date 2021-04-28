; RUN: opt < %s -loop-vectorize -mtriple aarch64-unknown-linux-gnu -mattr=+sve -enable-strict-reductions -S | FileCheck %s -check-prefix=CHECK

define float @fadd_strict(float* noalias nocapture readonly %a, i64 %n) {
; CHECK-LABEL: @fadd_strict
; CHECK: vector.body:
; CHECK: %[[VEC_PHI:.*]] = phi float [ 0.000000e+00, %vector.ph ], [ %[[RDX:.*]], %vector.body ]
; CHECK: %[[LOAD:.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK: %[[RDX]] = call float @llvm.vector.reduce.fadd.nxv8f32(float %[[VEC_PHI]], <vscale x 8 x float> %[[LOAD]])
; CHECK: for.end
; CHECK: %[[PHI:.*]] = phi float [ %[[SCALAR:.*]], %for.body ], [ %[[RDX]], %middle.block ]
; CHECK: ret float %[[PHI]]
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi float [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %a, i64 %iv
  %0 = load float, float* %arrayidx, align 4
  %add = fadd float %0, %sum.07
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret float %add
}

define float @fadd_strict_unroll(float* noalias nocapture readonly %a, i64 %n) {
; CHECK-LABEL: @fadd_strict_unroll
; CHECK: vector.body:
; CHECK: %[[VEC_PHI1:.*]] = phi float [ 0.000000e+00, %vector.ph ], [ %[[RDX4:.*]], %vector.body ]
; CHECK-NOT: phi float [ 0.000000e+00, %vector.ph ], [ %[[RDX4]], %vector.body ]
; CHECK: %[[LOAD1:.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK: %[[LOAD2:.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK: %[[LOAD3:.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK: %[[LOAD4:.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK: %[[RDX1:.*]] = call float @llvm.vector.reduce.fadd.nxv8f32(float %[[VEC_PHI1]], <vscale x 8 x float> %[[LOAD1]])
; CHECK: %[[RDX2:.*]] = call float @llvm.vector.reduce.fadd.nxv8f32(float %[[RDX1]], <vscale x 8 x float> %[[LOAD2]])
; CHECK: %[[RDX3:.*]] = call float @llvm.vector.reduce.fadd.nxv8f32(float %[[RDX2]], <vscale x 8 x float> %[[LOAD3]])
; CHECK: %[[RDX4]] = call float @llvm.vector.reduce.fadd.nxv8f32(float %[[RDX3]], <vscale x 8 x float> %[[LOAD4]])
; CHECK: for.end
; CHECK: %[[PHI:.*]] = phi float [ %[[SCALAR:.*]], %for.body ], [ %[[RDX4]], %middle.block ]
; CHECK: ret float %[[PHI]]
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi float [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %a, i64 %iv
  %0 = load float, float* %arrayidx, align 4
  %add = fadd float %0, %sum.07
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !1

for.end:
  ret float %add
}

define void @fadd_strict_interleave(float* noalias nocapture readonly %a, float* noalias nocapture readonly %b, i64 %n) {
; CHECK-LABEL: @fadd_strict_interleave
; CHECK: entry
; CHECK: %[[ARRAYIDX:.*]] = getelementptr inbounds float, float* %a, i64 1
; CHECK: %[[LOAD1:.*]] = load float, float* %a
; CHECK: %[[LOAD2:.*]] = load float, float* %[[ARRAYIDX]]
; CHECK: vector.ph
; CHECK: %[[STEPVEC1:.*]] = call <vscale x 4 x i64> @llvm.experimental.stepvector.nxv4i64()
; CHECK: %[[STEPVEC_ADD1:.*]] = add <vscale x 4 x i64> %[[STEPVEC1]], shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> poison, i64 0, i32 0), <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK: %[[STEPVEC_MUL:.*]] = mul <vscale x 4 x i64> %[[STEPVEC_ADD1]], shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> poison, i64 2, i32 0), <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK: %[[INDUCTION:.*]] = add <vscale x 4 x i64> shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> poison, i64 0, i32 0), <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer), %[[STEPVEC_MUL]]
; CHECK: vector.body
; CHECK: %[[VEC_PHI2:.*]] = phi float [ %[[LOAD2]], %vector.ph ], [ %[[RDX2:.*]], %vector.body ]
; CHECK: %[[VEC_PHI1:.*]] = phi float [ %[[LOAD1]], %vector.ph ], [ %[[RDX1:.*]], %vector.body ]
; CHECK: %[[VEC_IND:.*]] = phi <vscale x 4 x i64> [ %[[INDUCTION]], %vector.ph ], [ {{.*}}, %vector.body ]
; CHECK: %[[GEP1:.*]] = getelementptr inbounds float, float* %b, <vscale x 4 x i64> %[[VEC_IND]]
; CHECK: %[[MGATHER1:.*]] = call <vscale x 4 x float> @llvm.masked.gather.nxv4f32.nxv4p0f32(<vscale x 4 x float*> %[[GEP1]], i32 4, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> undef, i1 true, i32 0), <vscale x 4 x i1> undef, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> undef)
; CHECK: %[[RDX1]] = call float @llvm.vector.reduce.fadd.nxv4f32(float %[[VEC_PHI1]], <vscale x 4 x float> %[[MGATHER1]])
; CHECK: %[[OR:.*]] = or <vscale x 4 x i64> %[[VEC_IND]], shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> poison, i64 1, i32 0), <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK: %[[GEP2:.*]] = getelementptr inbounds float, float* %b, <vscale x 4 x i64> %[[OR]]
; CHECK: %[[MGATHER2:.*]] = call <vscale x 4 x float> @llvm.masked.gather.nxv4f32.nxv4p0f32(<vscale x 4 x float*> %[[GEP2]], i32 4, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> undef, i1 true, i32 0), <vscale x 4 x i1> undef, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> undef)
; CHECK: %[[RDX2]] = call float @llvm.vector.reduce.fadd.nxv4f32(float %[[VEC_PHI2]], <vscale x 4 x float> %[[MGATHER2]])
; CHECK: for.end
; CHECK ret void
entry:
  %arrayidxa = getelementptr inbounds float, float* %a, i64 1
  %a1 = load float, float* %a, align 4
  %a2 = load float, float* %arrayidxa, align 4
  br label %for.body

for.body:
  %add.phi1 = phi float [ %a2, %entry ], [ %add2, %for.body ]
  %add.phi2 = phi float [ %a1, %entry ], [ %add1, %for.body ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidxb1 = getelementptr inbounds float, float* %b, i64 %iv
  %0 = load float, float* %arrayidxb1, align 4
  %add1 = fadd float %0, %add.phi2
  %or = or i64 %iv, 1
  %arrayidxb2 = getelementptr inbounds float, float* %b, i64 %or
  %1 = load float, float* %arrayidxb2, align 4
  %add2 = fadd float %1, %add.phi1
  %iv.next = add nuw nsw i64 %iv, 2
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !2

for.end:
  store float %add1, float* %a, align 4
  store float %add2, float* %arrayidxa, align 4
  ret void
}

define float @fadd_invariant(float* noalias nocapture readonly %a, float* noalias nocapture readonly %b, i64 %n) {
; CHECK-LABEL: @fadd_invariant
; CHECK: vector.body
; CHECK: %[[VEC_PHI1:.*]] = phi float [ 0.000000e+00, %vector.ph ], [ %[[RDX:.*]], %vector.body ]
; CHECK: %[[LOAD1:.*]] = load <vscale x 4 x float>, <vscale x 4 x float>*
; CHECK: %[[LOAD2:.*]] = load <vscale x 4 x float>, <vscale x 4 x float>*
; CHECK: %[[ADD:.*]] = fadd <vscale x 4 x float> %[[LOAD1]], %[[LOAD2]]
; CHECK: %[[RDX]] = call float @llvm.vector.reduce.fadd.nxv4f32(float %[[VEC_PHI1]], <vscale x 4 x float> %[[ADD]])
; CHECK: for.end.loopexit
; CHECK: %[[EXIT_PHI:.*]] = phi float [ {{.*}}, %for.body ], [ %[[RDX]], %middle.block ]
; CHECK: for.end
; CHECK: %[[PHI:.*]] = phi float [ 0.000000e+00, %entry ], [ %[[EXIT_PHI]], %for.end.loopexit ]
; CHECK: ret float %[[PHI]]
entry:
  %arrayidx = getelementptr inbounds float, float* %a, i64 1
  %0 = load float, float* %arrayidx, align 4
  %cmp1 = fcmp ogt float %0, 5.000000e-01
  br i1 %cmp1, label %for.body, label %for.end

for.body:                                      ; preds = %for.body
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %res.014 = phi float [ 0.000000e+00, %entry ], [ %rdx, %for.body ]
  %arrayidx2 = getelementptr inbounds float, float* %a, i64 %iv
  %1 = load float, float* %arrayidx2, align 4
  %arrayidx4 = getelementptr inbounds float, float* %b, i64 %iv
  %2 = load float, float* %arrayidx4, align 4
  %add = fadd float %1, %2
  %rdx = fadd float %res.014, %add
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !2

for.end:                                 ; preds = %for.body, %entry
  %res = phi float [ 0.000000e+00, %entry ], [ %rdx, %for.body ]
  ret float %res
}

define float @fadd_conditional(float* noalias nocapture readonly %a, float* noalias nocapture readonly %b, i64 %n) {
; CHECK-LABEL: @fadd_conditional
; CHECK: vector.body
; CHECK: %[[VEC_PHI:.*]] = phi float [ 1.000000e+00, %vector.ph ], [ %[[RDX:.*]], %vector.body ]
; CHECK: %[[LOAD:.*]] = load <vscale x 4 x float>, <vscale x 4 x float>*
; CHECK: %[[FCMP:.*]] = fcmp une <vscale x 4 x float> %[[LOAD]], shufflevector (<vscale x 4 x float> insertelement (<vscale x 4 x float> poison, float 0.000000e+00, i32 0), <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK: %[[MASKED_LOAD:.*]] = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0nxv4f32(<vscale x 4 x float>* {{.*}}, i32 4, <vscale x 4 x i1> %[[FCMP]], <vscale x 4 x float> poison)
; CHECK: %[[XOR:.*]] = xor <vscale x 4 x i1> %[[FCMP]], shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> undef, i1 true, i32 0), <vscale x 4 x i1> undef, <vscale x 4 x i32> zeroinitializer)
; CHECK: %[[SELECT:.*]] = select <vscale x 4 x i1> %[[XOR]], <vscale x 4 x float> shufflevector (<vscale x 4 x float> insertelement (<vscale x 4 x float> poison, float 3.000000e+00, i32 0), <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %[[MASKED_LOAD]]
; CHECK: %[[RDX]] = call float @llvm.vector.reduce.fadd.nxv4f32(float %[[VEC_PHI]], <vscale x 4 x float> %[[SELECT]])
; CHECK: scalar.ph
; CHECK: %[[MERGE_RDX:.*]] = phi float [ 1.000000e+00, %entry ], [ %[[RDX]], %middle.block ]
; CHECK: for.body
; CHECK: %[[RES:.*]] = phi float [ %[[MERGE_RDX]], %scalar.ph ], [ %[[FADD:.*]], %for.inc ]
; CHECK: if.then
; CHECK: %[[LOAD2:.*]] = load float, float*
; CHECK: for.inc
; CHECK: %[[PHI:.*]] = phi float [ %[[LOAD2]], %if.then ], [ 3.000000e+00, %for.body ]
; CHECK: %[[FADD]] = fadd float %[[RES]], %[[PHI]]
; CHECK: for.end
; CHECK: %[[RDX_PHI:.*]] = phi float [ %[[FADD]], %for.inc ], [ %[[RDX]], %middle.block ]
; CHECK: ret float %[[RDX_PHI]]
entry:
  br label %for.body

for.body:                                      ; preds = %for.body
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %res = phi float [ 1.000000e+00, %entry ], [ %fadd, %for.inc ]
  %arrayidx = getelementptr inbounds float, float* %b, i64 %iv
  %0 = load float, float* %arrayidx, align 4
  %tobool = fcmp une float %0, 0.000000e+00
  br i1 %tobool, label %if.then, label %for.inc

if.then:                                      ; preds = %for.body
  %arrayidx2 = getelementptr inbounds float, float* %a, i64 %iv
  %1 = load float, float* %arrayidx2, align 4
  br label %for.inc

for.inc:
  %phi = phi float [ %1, %if.then ], [ 3.000000e+00, %for.body ]
  %fadd = fadd float %res, %phi
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !2

for.end:
  %rdx = phi float [ %fadd, %for.inc ]
  ret float %rdx
}

; Negative test - loop contains multiple fadds which we cannot safely reorder
; Note: This test vectorizes the loop with a non-strict implementation, which reorders the FAdd operations.
; This is happening because we are using hints, where allowReordering returns true.
define float @fadd_multiple(float* noalias nocapture %a, float* noalias nocapture %b, i64 %n) {
; CHECK-LABEL: @fadd_multiple
; CHECK: vector.body
; CHECK: %[[PHI:.*]] = phi <vscale x 8 x float> [ insertelement (<vscale x 8 x float> shufflevector (<vscale x 8 x float> insertelement (<vscale x 8 x float> undef, float -0.000000e+00, i32 0), <vscale x 8 x float> undef, <vscale x 8 x i32> zeroinitializer), float -0.000000e+00, i32 0), %vector.ph ], [ %[[VEC_FADD2:.*]], %vector.body ]
; CHECK: %[[VEC_LOAD1:.*]] = load <vscale x 8 x float>, <vscale x 8 x float>
; CHECK: %[[VEC_FADD1:.*]] = fadd <vscale x 8 x float> %[[PHI]], %[[VEC_LOAD1]]
; CHECK: %[[VEC_LOAD2:.*]] = load <vscale x 8 x float>, <vscale x 8 x float>
; CHECK: %[[VEC_FADD2]] = fadd <vscale x 8 x float> %[[VEC_FADD1]], %[[VEC_LOAD2]]
; CHECK: middle.block
; CHECK: %[[RDX:.*]] = call float @llvm.vector.reduce.fadd.nxv8f32(float -0.000000e+00, <vscale x 8 x float> %[[VEC_FADD2]])
; CHECK: for.body
; CHECK: %[[SUM:.*]] = phi float [ %bc.merge.rdx, %scalar.ph ], [ %[[FADD2:.*]], %for.body ]
; CHECK: %[[LOAD1:.*]] = load float, float*
; CHECK: %[[FADD1:.*]] = fadd float %[[SUM]], %[[LOAD1]]
; CHECK: %[[LOAD2:.*]] = load float, float*
; CHECK: %[[FADD2]] = fadd float %[[FADD1]], %[[LOAD2]]
; CHECK: for.end
; CHECK: %[[RET:.*]] = phi float [ %[[FADD2]], %for.body ], [ %[[RDX]], %middle.block ]
; CHECK: ret float %[[RET]]
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum = phi float [ -0.000000e+00, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %a, i64 %iv
  %0 = load float, float* %arrayidx, align 4
  %add = fadd float %sum, %0
  %arrayidx2 = getelementptr inbounds float, float* %b, i64 %iv
  %1 = load float, float* %arrayidx2, align 4
  %add3 = fadd float %add, %1
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:                                         ; preds = %for.body
  %rdx = phi float [ %add3, %for.body ]
  ret float %rdx
}

!0 = distinct !{!0, !3, !6, !8}
!1 = distinct !{!1, !3, !7, !8}
!2 = distinct !{!2, !4, !6, !8}
!3 = !{!"llvm.loop.vectorize.width", i32 8}
!4 = !{!"llvm.loop.vectorize.width", i32 4}
!5 = !{!"llvm.loop.vectorize.width", i32 2}
!6 = !{!"llvm.loop.interleave.count", i32 1}
!7 = !{!"llvm.loop.interleave.count", i32 4}
!8 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
