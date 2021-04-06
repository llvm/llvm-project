; RUN: opt < %s -loop-vectorize -instcombine -mtriple aarch64-unknown-linux-gnu -enable-strict-reductions -S | FileCheck %s -check-prefix=CHECK

define float @fadd_strict(float* noalias nocapture readonly %a, i64 %n) {
; CHECK-LABEL: @fadd_strict
; CHECK: vector.body:
; CHECK: %[[VEC_PHI:.*]] = phi float [ 0.000000e+00, %vector.ph ], [ %[[RDX:.*]], %vector.body ]
; CHECK: %[[LOAD:.*]] = load <8 x float>, <8 x float>*
; CHECK: %[[RDX]] = call float @llvm.vector.reduce.fadd.v8f32(float %[[VEC_PHI]], <8 x float> %[[LOAD]])
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
; CHECK: %[[LOAD1:.*]] = load <8 x float>, <8 x float>*
; CHECK: %[[LOAD2:.*]] = load <8 x float>, <8 x float>*
; CHECK: %[[LOAD3:.*]] = load <8 x float>, <8 x float>*
; CHECK: %[[LOAD4:.*]] = load <8 x float>, <8 x float>*
; CHECK: %[[RDX1:.*]] = call float @llvm.vector.reduce.fadd.v8f32(float %[[VEC_PHI1]], <8 x float> %[[LOAD1]])
; CHECK: %[[RDX2:.*]] = call float @llvm.vector.reduce.fadd.v8f32(float %[[RDX1]], <8 x float> %[[LOAD2]])
; CHECK: %[[RDX3:.*]] = call float @llvm.vector.reduce.fadd.v8f32(float %[[RDX2]], <8 x float> %[[LOAD3]])
; CHECK: %[[RDX4:.*]] = call float @llvm.vector.reduce.fadd.v8f32(float %[[RDX3]], <8 x float> %[[LOAD4]])
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
; CHECK: vector.body
; CHECK: %[[VEC_PHI1:.*]] = phi float [ %[[LOAD2]], %vector.ph ], [ %[[RDX2:.*]], %vector.body ]
; CHECK: %[[VEC_PHI2:.*]] = phi float [ %[[LOAD1]], %vector.ph ], [ %[[RDX1:.*]], %vector.body ]
; CHECK: %[[WIDE_LOAD:.*]] = load <8 x float>, <8 x float>*
; CHECK: %[[STRIDED1:.*]] = shufflevector <8 x float> %[[WIDE_LOAD]], <8 x float> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK: %[[STRIDED2:.*]] = shufflevector <8 x float> %[[WIDE_LOAD]], <8 x float> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
; CHECK: %[[RDX1]] = call float @llvm.vector.reduce.fadd.v4f32(float %[[VEC_PHI2]], <4 x float> %[[STRIDED1]])
; CHECK: %[[RDX2]] = call float @llvm.vector.reduce.fadd.v4f32(float %[[VEC_PHI1]], <4 x float> %[[STRIDED2]])
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
; CHECK: %[[LOAD1:.*]] = load <4 x float>, <4 x float>*
; CHECK: %[[LOAD2:.*]] = load <4 x float>, <4 x float>*
; CHECK: %[[ADD:.*]] = fadd <4 x float> %[[LOAD1]], %[[LOAD2]]
; CHECK: %[[RDX]] = call float @llvm.vector.reduce.fadd.v4f32(float %[[VEC_PHI1]], <4 x float> %[[ADD]])
; CHECK: for.end.loopexit
; CHECK: %[[EXIT_PHI:.*]] = phi float [ %[[SCALAR:.*]], %for.body ], [ %[[RDX]], %middle.block ]
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
; CHECK: vector.body:
; CHECK: %[[PHI:.*]] = phi float [ 1.000000e+00, %vector.ph ], [ %[[RDX:.*]], %pred.load.continue6 ]
; CHECK: %[[LOAD1:.*]] = load <4 x float>, <4 x float>*
; CHECK: %[[FCMP1:.*]] = fcmp une <4 x float> %[[LOAD1]], zeroinitializer
; CHECK: %[[EXTRACT:.*]] = extractelement <4 x i1> %[[FCMP1]], i32 0
; CHECK: br i1 %[[EXTRACT]], label %pred.load.if, label %pred.load.continue
; CHECK: pred.load.continue6
; CHECK: %[[PHI1:.*]] = phi <4 x float> [ %[[PHI0:.*]], %pred.load.continue4 ], [ %[[INS_ELT:.*]], %pred.load.if5 ]
; CHECK: %[[PRED:.*]] = select <4 x i1> %[[FCMP1]], <4 x float> %[[PHI1]], <4 x float> <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
; CHECK: %[[RDX]] = call float @llvm.vector.reduce.fadd.v4f32(float %[[PHI]], <4 x float> %[[PRED]])
; CHECK: for.body
; CHECK: %[[RES_PHI:.*]] = phi float [ %[[MERGE_RDX:.*]], %scalar.ph ], [ %[[FADD:.*]], %for.inc ]
; CHECK: %[[LOAD2:.*]] = load float, float*
; CHECK: %[[FCMP2:.*]] = fcmp une float %[[LOAD2]], 0.000000e+00
; CHECK: br i1 %[[FCMP2]], label %if.then, label %for.inc
; CHECK: if.then
; CHECK: %[[LOAD3:.*]] = load float, float*
; CHECK: br label %for.inc
; CHECK: for.inc
; CHECK: %[[PHI2:.*]] = phi float [ %[[LOAD3]], %if.then ], [ 3.000000e+00, %for.body ]
; CHECK: %[[FADD]] = fadd float %[[RES_PHI]], %[[PHI2]]
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

; Test to check masking correct, using the "llvm.loop.vectorize.predicate.enable" attribute
define float @fadd_predicated(float* noalias nocapture %a, i64 %n) {
; CHECK-LABEL: @fadd_predicated
; CHECK: vector.ph
; CHECK: %[[TRIP_MINUS_ONE:.*]] = add i64 %n, -1
; CHECK: %[[BROADCAST_INS:.*]] = insertelement <2 x i64> poison, i64 %[[TRIP_MINUS_ONE]], i32 0
; CHECK: %[[SPLAT:.*]] = shufflevector <2 x i64> %[[BROADCAST_INS]], <2 x i64> poison, <2 x i32> zeroinitializer
; CHECK: vector.body
; CHECK: %[[RDX_PHI:.*]] =  phi float [ 0.000000e+00, %vector.ph ], [ %[[RDX:.*]], %pred.load.continue2 ]
; CHECK: pred.load.continue2
; CHECK: %[[PHI:.*]] = phi <2 x float> [ %[[PHI0:.*]], %pred.load.continue ], [ %[[INS_ELT:.*]], %pred.load.if1 ]
; CHECK: %[[MASK:.*]] = select <2 x i1> %0, <2 x float> %[[PHI]], <2 x float> <float -0.000000e+00, float -0.000000e+00>
; CHECK: %[[RDX]] = call float @llvm.vector.reduce.fadd.v2f32(float %[[RDX_PHI]], <2 x float> %[[MASK]])
; CHECK: for.end:
; CHECK: %[[RES_PHI:.*]] = phi float [ undef, %for.body ], [ %[[RDX]], %middle.block ]
; CHECK: ret float %[[RES_PHI]]
entry:
  br label %for.body

for.body:                                           ; preds = %entry, %for.body
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %sum.02 = phi float [ %l7, %for.body ], [ 0.000000e+00, %entry ]
  %l2 = getelementptr inbounds float, float* %a, i64 %iv
  %l3 = load float, float* %l2, align 4
  %l7 = fadd float %sum.02, %l3
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !3

for.end:                                            ; preds = %for.body
  %sum.0.lcssa = phi float [ %l7, %for.body ]
  ret float %sum.0.lcssa
}

; Negative test - loop contains multiple fadds which we cannot safely reorder
define float @fadd_multiple(float* noalias nocapture %a, float* noalias nocapture %b, i64 %n) {
; CHECK-LABEL: @fadd_multiple
; CHECK: vector.body
; CHECK: %[[PHI:.*]] = phi <8 x float> [ <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %vector.ph ], [ %[[VEC_FADD2:.*]], %vector.body ]
; CHECK: %[[VEC_LOAD1:.*]] = load <8 x float>, <8 x float>
; CHECK: %[[VEC_FADD1:.*]] = fadd <8 x float> %[[PHI]], %[[VEC_LOAD1]]
; CHECK: %[[VEC_LOAD2:.*]] = load <8 x float>, <8 x float>
; CHECK: %[[VEC_FADD2]] = fadd <8 x float> %[[VEC_FADD1]], %[[VEC_LOAD2]]
; CHECK: middle.block
; CHECK: %[[RDX:.*]] = call float @llvm.vector.reduce.fadd.v8f32(float -0.000000e+00, <8 x float> %[[VEC_FADD2]])
; CHECK: for.body
; CHECK: %[[SUM:.*]] = phi float [ %bc.merge.rdx, %scalar.ph ], [ %[[FADD2:.*]], %for.body ]
; CHECK: %[[LOAD1:.*]] = load float, float*
; CHECK: %[[FADD1:.*]] = fadd float %sum, %[[LOAD1]]
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

!0 = distinct !{!0, !4, !7, !9}
!1 = distinct !{!1, !4, !8, !9}
!2 = distinct !{!2, !5, !7, !9}
!3 = distinct !{!3, !6, !7, !9, !10}
!4 = !{!"llvm.loop.vectorize.width", i32 8}
!5 = !{!"llvm.loop.vectorize.width", i32 4}
!6 = !{!"llvm.loop.vectorize.width", i32 2}
!7 = !{!"llvm.loop.interleave.count", i32 1}
!8 = !{!"llvm.loop.interleave.count", i32 4}
!9 = !{!"llvm.loop.vectorize.enable", i1 true}
!10 = !{!"llvm.loop.vectorize.predicate.enable", i1 true}
