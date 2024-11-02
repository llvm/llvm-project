; RUN: opt < %s -loop-vectorize -mtriple aarch64-unknown-linux-gnu -force-ordered-reductions=false -hints-allow-reordering=false -S 2>%t | FileCheck %s --check-prefix=CHECK-NOT-VECTORIZED
; RUN: opt < %s -loop-vectorize -mtriple aarch64-unknown-linux-gnu -force-ordered-reductions=false -hints-allow-reordering=true  -S 2>%t | FileCheck %s --check-prefix=CHECK-UNORDERED
; RUN: opt < %s -loop-vectorize -mtriple aarch64-unknown-linux-gnu -force-ordered-reductions=true  -hints-allow-reordering=false -S 2>%t | FileCheck %s --check-prefix=CHECK-ORDERED
; RUN: opt < %s -loop-vectorize -mtriple aarch64-unknown-linux-gnu -force-ordered-reductions=true  -hints-allow-reordering=true  -S 2>%t | FileCheck %s --check-prefix=CHECK-UNORDERED
; RUN: opt < %s -loop-vectorize -mtriple aarch64-unknown-linux-gnu -hints-allow-reordering=false -S 2>%t | FileCheck %s --check-prefix=CHECK-ORDERED

define float @fadd_strict(float* noalias nocapture readonly %a, i64 %n) {
; CHECK-ORDERED-LABEL: @fadd_strict
; CHECK-ORDERED: vector.body:
; CHECK-ORDERED: %[[VEC_PHI:.*]] = phi float [ 0.000000e+00, %vector.ph ], [ %[[RDX:.*]], %vector.body ]
; CHECK-ORDERED: %[[LOAD:.*]] = load <8 x float>, <8 x float>*
; CHECK-ORDERED: %[[RDX]] = call float @llvm.vector.reduce.fadd.v8f32(float %[[VEC_PHI]], <8 x float> %[[LOAD]])
; CHECK-ORDERED: for.end
; CHECK-ORDERED: %[[PHI:.*]] = phi float [ %[[SCALAR:.*]], %for.body ], [ %[[RDX]], %middle.block ]
; CHECK-ORDERED: ret float %[[PHI]]

; CHECK-UNORDERED-LABEL: @fadd_strict
; CHECK-UNORDERED: vector.body
; CHECK-UNORDERED: %[[VEC_PHI:.*]] = phi <8 x float> [ <float 0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %vector.ph ], [ %[[FADD_VEC:.*]], %vector.body ]
; CHECK-UNORDERED: %[[LOAD_VEC:.*]] = load <8 x float>, <8 x float>*
; CHECK-UNORDERED: %[[FADD_VEC]] = fadd <8 x float> %[[LOAD_VEC]], %[[VEC_PHI]]
; CHECK-UNORDERED-NOT: call float @llvm.vector.reduce.fadd
; CHECK-UNORDERED: middle.block
; CHECK-UNORDERED: %[[RDX:.*]] = call float @llvm.vector.reduce.fadd.v8f32(float -0.000000e+00, <8 x float> %[[FADD_VEC]])
; CHECK-UNORDERED: for.body
; CHECK-UNORDERED: %[[LOAD:.*]] = load float, float*
; CHECK-UNORDERED: %[[FADD:.*]] = fadd float %[[LOAD]], {{.*}}
; CHECK-UNORDERED: for.end
; CHECK-UNORDERED: %[[RES:.*]] = phi float [ %[[FADD]], %for.body ], [ %[[RDX]], %middle.block ]
; CHECK-UNORDERED: ret float %[[RES]]

; CHECK-NOT-VECTORIZED-LABEL: @fadd_strict
; CHECK-NOT-VECTORIZED-NOT: vector.body

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

; Same as above but where fadd has a fast-math flag.
define float @fadd_strict_fmf(float* noalias nocapture readonly %a, i64 %n) {
; CHECK-ORDERED-LABEL: @fadd_strict_fmf
; CHECK-ORDERED: vector.body:
; CHECK-ORDERED: [[VEC_PHI:%.*]] = phi float [ 0.000000e+00, %vector.ph ], [ [[RDX:%.*]], %vector.body ]
; CHECK-ORDERED: [[LOAD_VEC:%.*]] = load <8 x float>, <8 x float>*
; CHECK-ORDERED: [[RDX]] = call nnan float @llvm.vector.reduce.fadd.v8f32(float [[VEC_PHI]], <8 x float> [[LOAD_VEC]])
; CHECK-ORDERED: for.end:
; CHECK-ORDERED: [[RES:%.*]] = phi float [ [[SCALAR:%.*]], %for.body ], [ [[RDX]], %middle.block ]
; CHECK-ORDERED: ret float [[RES]]

; CHECK-UNORDERED-LABEL: @fadd_strict_fmf
; CHECK-UNORDERED: vector.body:
; CHECK-UNORDERED: [[VEC_PHI:%.*]] = phi <8 x float> [ <float 0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %vector.ph ], [ [[FADD_VEC:%.*]], %vector.body ]
; CHECK-UNORDERED: [[LOAD_VEC:%.*]] = load <8 x float>, <8 x float>*
; CHECK-UNORDERED: [[FADD_VEC]] = fadd nnan <8 x float> [[LOAD_VEC]], [[VEC_PHI]]
; CHECK-UNORDERED-NOT: @llvm.vector.reduce.fadd 
; CHECK-UNORDERED: middle.block:
; CHECK-UNORDERED: [[RDX:%.*]] = call nnan float @llvm.vector.reduce.fadd.v8f32(float -0.000000e+00, <8 x float> [[FADD_VEC]])
; CHECK-UNORDERED: for.body:
; CHECK-UNORDERED: [[LOAD:%.*]] = load float, float*
; CHECK-UNORDERED: [[FADD:%.*]] = fadd nnan float [[LOAD]], {{.*}}
; CHECK-UNORDERED: for.end:
; CHECK-UNORDERED: [[RES:%.*]] = phi float [ [[FADD]], %for.body ], [ [[RDX]], %middle.block ]
; CHECK-UNORDERED: ret float [[RES]]

; CHECK-NOT-VECTORIZED-LABEL: @fadd_strict_fmf
; CHECK-NOT-VECTORIZED-NOT: vector.body

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi float [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %a, i64 %iv
  %0 = load float, float* %arrayidx, align 4
  %add = fadd nnan float %0, %sum.07
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret float %add
}

define float @fadd_strict_unroll(float* noalias nocapture readonly %a, i64 %n) {
; CHECK-ORDERED-LABEL: @fadd_strict_unroll
; CHECK-ORDERED: vector.body:
; CHECK-ORDERED: %[[VEC_PHI1:.*]] = phi float [ 0.000000e+00, %vector.ph ], [ %[[RDX4:.*]], %vector.body ]
; CHECK-ORDERED-NOT: phi float [ 0.000000e+00, %vector.ph ], [ %[[RDX4]], %vector.body ]
; CHECK-ORDERED: %[[LOAD1:.*]] = load <8 x float>, <8 x float>*
; CHECK-ORDERED: %[[LOAD2:.*]] = load <8 x float>, <8 x float>*
; CHECK-ORDERED: %[[LOAD3:.*]] = load <8 x float>, <8 x float>*
; CHECK-ORDERED: %[[LOAD4:.*]] = load <8 x float>, <8 x float>*
; CHECK-ORDERED: %[[RDX1:.*]] = call float @llvm.vector.reduce.fadd.v8f32(float %[[VEC_PHI1]], <8 x float> %[[LOAD1]])
; CHECK-ORDERED: %[[RDX2:.*]] = call float @llvm.vector.reduce.fadd.v8f32(float %[[RDX1]], <8 x float> %[[LOAD2]])
; CHECK-ORDERED: %[[RDX3:.*]] = call float @llvm.vector.reduce.fadd.v8f32(float %[[RDX2]], <8 x float> %[[LOAD3]])
; CHECK-ORDERED: %[[RDX4]] = call float @llvm.vector.reduce.fadd.v8f32(float %[[RDX3]], <8 x float> %[[LOAD4]])
; CHECK-ORDERED: for.end
; CHECK-ORDERED: %[[PHI:.*]] = phi float [ %[[SCALAR:.*]], %for.body ], [ %[[RDX4]], %middle.block ]
; CHECK-ORDERED: ret float %[[PHI]]

; CHECK-UNORDERED-LABEL: @fadd_strict_unroll
; CHECK-UNORDERED: vector.body
; CHECK-UNORDERED:  %[[VEC_PHI1:.*]] = phi <8 x float> [ <float 0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %vector.ph ], [ %[[VEC_FADD1:.*]], %vector.body ]
; CHECK-UNORDERED:  %[[VEC_PHI2:.*]] = phi <8 x float> [ <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %vector.ph ], [ %[[VEC_FADD2:.*]], %vector.body ]
; CHECK-UNORDERED:  %[[VEC_PHI3:.*]] = phi <8 x float> [ <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %vector.ph ], [ %[[VEC_FADD3:.*]], %vector.body ]
; CHECK-UNORDERED:  %[[VEC_PHI4:.*]] = phi <8 x float> [ <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %vector.ph ], [ %[[VEC_FADD4:.*]], %vector.body ]
; CHECK-UNORDERED: %[[VEC_LOAD1:.*]] = load <8 x float>, <8 x float>*
; CHECK-UNORDERED: %[[VEC_LOAD2:.*]] = load <8 x float>, <8 x float>*
; CHECK-UNORDERED: %[[VEC_LOAD3:.*]] = load <8 x float>, <8 x float>*
; CHECK-UNORDERED: %[[VEC_LOAD4:.*]] = load <8 x float>, <8 x float>*
; CHECK-UNORDERED: %[[VEC_FADD1]] = fadd <8 x float> %[[VEC_LOAD1]], %[[VEC_PHI1]]
; CHECK-UNORDERED: %[[VEC_FADD2]] = fadd <8 x float> %[[VEC_LOAD2]], %[[VEC_PHI2]]
; CHECK-UNORDERED: %[[VEC_FADD3]] = fadd <8 x float> %[[VEC_LOAD3]], %[[VEC_PHI3]]
; CHECK-UNORDERED: %[[VEC_FADD4]] = fadd <8 x float> %[[VEC_LOAD4]], %[[VEC_PHI4]]
; CHECK-UNORDERED-NOT: call float @llvm.vector.reduce.fadd
; CHECK-UNORDERED: middle.block
; CHECK-UNORDERED: %[[BIN_RDX1:.*]] = fadd <8 x float> %[[VEC_FADD2]], %[[VEC_FADD1]]
; CHECK-UNORDERED: %[[BIN_RDX2:.*]] = fadd <8 x float> %[[VEC_FADD3]], %[[BIN_RDX1]]
; CHECK-UNORDERED: %[[BIN_RDX3:.*]] = fadd <8 x float> %[[VEC_FADD4]], %[[BIN_RDX2]]
; CHECK-UNORDERED: %[[RDX:.*]] = call float @llvm.vector.reduce.fadd.v8f32(float -0.000000e+00, <8 x float> %[[BIN_RDX3]])
; CHECK-UNORDERED: for.body
; CHECK-UNORDERED: %[[LOAD:.*]] = load float, float*
; CHECK-UNORDERED: %[[FADD:.*]] = fadd float %[[LOAD]], {{.*}}
; CHECK-UNORDERED: for.end
; CHECK-UNORDERED: %[[RES:.*]] = phi float [ %[[FADD]], %for.body ], [ %[[RDX]], %middle.block ]
; CHECK-UNORDERED: ret float %[[RES]]

; CHECK-NOT-VECTORIZED-LABEL: @fadd_strict_unroll
; CHECK-NOT-VECTORIZED-NOT: vector.body

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

; An additional test for unrolling where we need the last value of the reduction, i.e:
; float sum = 0, sum2;
; for(int i=0; i<N; ++i) {
;   sum += ptr[i];
;   *ptr2 = sum + 42;
; }
; return sum;

define float @fadd_strict_unroll_last_val(float* noalias nocapture readonly %a, float* noalias nocapture readonly %b, i64 %n) {
; CHECK-ORDERED-LABEL: @fadd_strict_unroll_last_val
; CHECK-ORDERED: vector.body
; CHECK-ORDERED: %[[VEC_PHI1:.*]] = phi float [ 0.000000e+00, %vector.ph ], [ %[[RDX4:.*]], %vector.body ]
; CHECK-ORDERED-NOT: phi float [ 0.000000e+00, %vector.ph ], [ %[[RDX4]], %vector.body ]
; CHECK-ORDERED: %[[LOAD1:.*]] = load <8 x float>, <8 x float>*
; CHECK-ORDERED: %[[LOAD2:.*]] = load <8 x float>, <8 x float>*
; CHECK-ORDERED: %[[LOAD3:.*]] = load <8 x float>, <8 x float>*
; CHECK-ORDERED: %[[LOAD4:.*]] = load <8 x float>, <8 x float>*
; CHECK-ORDERED: %[[RDX1:.*]] = call float @llvm.vector.reduce.fadd.v8f32(float %[[VEC_PHI1]], <8 x float> %[[LOAD1]])
; CHECK-ORDERED: %[[RDX2:.*]] = call float @llvm.vector.reduce.fadd.v8f32(float %[[RDX1]], <8 x float> %[[LOAD2]])
; CHECK-ORDERED: %[[RDX3:.*]] = call float @llvm.vector.reduce.fadd.v8f32(float %[[RDX2]], <8 x float> %[[LOAD3]])
; CHECK-ORDERED: %[[RDX4]] = call float @llvm.vector.reduce.fadd.v8f32(float %[[RDX3]], <8 x float> %[[LOAD4]])
; CHECK-ORDERED: for.body
; CHECK-ORDERED: %[[SUM_PHI:.*]] = phi float [ %[[FADD:.*]], %for.body ], [ {{.*}}, %scalar.ph ]
; CHECK-ORDERED: %[[LOAD5:.*]] = load float, float*
; CHECK-ORDERED: %[[FADD]] =  fadd float %[[SUM_PHI]], %[[LOAD5]]
; CHECK-ORDERED: for.cond.cleanup
; CHECK-ORDERED: %[[FADD_LCSSA:.*]] = phi float [ %[[FADD]], %for.body ], [ %[[RDX4]], %middle.block ]
; CHECK-ORDERED: %[[FADD_42:.*]] = fadd float %[[FADD_LCSSA]], 4.200000e+01
; CHECK-ORDERED: store float %[[FADD_42]], float* %b
; CHECK-ORDERED: for.end
; CHECK-ORDERED: %[[SUM_LCSSA:.*]] = phi float [ %[[FADD_LCSSA]], %for.cond.cleanup ], [ 0.000000e+00, %entry ]
; CHECK-ORDERED: ret float %[[SUM_LCSSA]]

; CHECK-UNORDERED-LABEL: @fadd_strict_unroll_last_val
; CHECK-UNORDERED: vector.body
; CHECK-UNORDERED: %[[VEC_PHI1:.*]] = phi <8 x float> [ <float 0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %vector.ph ], [ %[[VEC_FADD1:.*]], %vector.body ]
; CHECK-UNORDERED: %[[VEC_PHI2:.*]] = phi <8 x float> [ <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %vector.ph ], [ %[[VEC_FADD2:.*]], %vector.body ]
; CHECK-UNORDERED: %[[VEC_PHI3:.*]] = phi <8 x float> [ <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %vector.ph ], [ %[[VEC_FADD3:.*]], %vector.body ]
; CHECK-UNORDERED: %[[VEC_PHI4:.*]] = phi <8 x float> [ <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %vector.ph ], [ %[[VEC_FADD4:.*]], %vector.body ]
; CHECK-UNORDERED: %[[VEC_LOAD1:.*]] = load <8 x float>, <8 x float>*
; CHECK-UNORDERED: %[[VEC_LOAD2:.*]] = load <8 x float>, <8 x float>*
; CHECK-UNORDERED: %[[VEC_LOAD3:.*]] = load <8 x float>, <8 x float>*
; CHECK-UNORDERED: %[[VEC_LOAD4:.*]] = load <8 x float>, <8 x float>*
; CHECK-UNORDERED: %[[VEC_FADD1]] = fadd <8 x float> %[[VEC_PHI1]], %[[VEC_LOAD1]]
; CHECK-UNORDERED: %[[VEC_FADD2]] = fadd <8 x float> %[[VEC_PHI2]], %[[VEC_LOAD2]]
; CHECK-UNORDERED: %[[VEC_FADD3]] = fadd <8 x float> %[[VEC_PHI3]], %[[VEC_LOAD3]]
; CHECK-UNORDERED: %[[VEC_FADD4]] = fadd <8 x float> %[[VEC_PHI4]], %[[VEC_LOAD4]]
; CHECK-UNORDERED-NOT: call float @llvm.vector.reduce.fadd
; CHECK-UNORDERED: middle.block
; CHECK-UNORDERED: %[[BIN_RDX1:.*]] = fadd <8 x float> %[[VEC_FADD2]], %[[VEC_FADD1]]
; CHECK-UNORDERED: %[[BIN_RDX2:.*]] = fadd <8 x float> %[[VEC_FADD3]], %[[BIN_RDX1]]
; CHECK-UNORDERED: %[[BIN_RDX3:.*]] = fadd <8 x float> %[[VEC_FADD4]], %[[BIN_RDX2]]
; CHECK-UNORDERED: %[[RDX:.*]] = call float @llvm.vector.reduce.fadd.v8f32(float -0.000000e+00, <8 x float> %[[BIN_RDX3]])
; CHECK-UNORDERED: for.body
; CHECK-UNORDERED: %[[LOAD:.*]] = load float, float*
; CHECK-UNORDERED: %[[FADD:.*]] = fadd float {{.*}}, %[[LOAD]]
; CHECK-UNORDERED: for.cond.cleanup
; CHECK-UNORDERED: %[[FADD_LCSSA:.*]] = phi float [ %[[FADD]], %for.body ], [ %[[RDX]], %middle.block ]
; CHECK-UNORDERED: %[[FADD_42:.*]] = fadd float %[[FADD_LCSSA]], 4.200000e+01
; CHECK-UNORDERED: store float %[[FADD_42]], float* %b
; CHECK-UNORDERED: for.end
; CHECK-UNORDERED: %[[SUM_LCSSA:.*]] = phi float [ %[[FADD_LCSSA]], %for.cond.cleanup ], [ 0.000000e+00, %entry ]
; CHECK-UNORDERED: ret float %[[SUM_LCSSA]]

; CHECK-NOT-VECTORIZED-LABEL: @fadd_strict_unroll_last_val
; CHECK-NOT-VECTORIZED-NOT: vector.body

entry:
  %cmp = icmp sgt i64 %n, 0
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum = phi float [ 0.000000e+00, %entry ], [ %fadd, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %a, i64 %iv
  %0 = load float, float* %arrayidx, align 4
  %fadd = fadd float %sum, %0
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !1

for.cond.cleanup:
  %fadd.lcssa = phi float [ %fadd, %for.body ]
  %fadd2 = fadd float %fadd.lcssa, 4.200000e+01
  store float %fadd2, float* %b, align 4
  br label %for.end

for.end:
  %sum.lcssa = phi float [ %fadd.lcssa, %for.cond.cleanup ], [ 0.000000e+00, %entry ]
  ret float %sum.lcssa
}

define void @fadd_strict_interleave(float* noalias nocapture readonly %a, float* noalias nocapture readonly %b, i64 %n) {
; CHECK-ORDERED-LABEL: @fadd_strict_interleave
; CHECK-ORDERED: entry
; CHECK-ORDERED: %[[ARRAYIDX:.*]] = getelementptr inbounds float, float* %a, i64 1
; CHECK-ORDERED: %[[LOAD1:.*]] = load float, float* %a
; CHECK-ORDERED: %[[LOAD2:.*]] = load float, float* %[[ARRAYIDX]]
; CHECK-ORDERED: vector.body
; CHECK-ORDERED: %[[VEC_PHI1:.*]] = phi float [ %[[LOAD2]], %vector.ph ], [ %[[RDX2:.*]], %vector.body ]
; CHECK-ORDERED: %[[VEC_PHI2:.*]] = phi float [ %[[LOAD1]], %vector.ph ], [ %[[RDX1:.*]], %vector.body ]
; CHECK-ORDERED: %[[WIDE_LOAD:.*]] = load <8 x float>, <8 x float>*
; CHECK-ORDERED: %[[STRIDED1:.*]] = shufflevector <8 x float> %[[WIDE_LOAD]], <8 x float> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK-ORDERED: %[[STRIDED2:.*]] = shufflevector <8 x float> %[[WIDE_LOAD]], <8 x float> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
; CHECK-ORDERED: %[[RDX2]] = call float @llvm.vector.reduce.fadd.v4f32(float %[[VEC_PHI1]], <4 x float> %[[STRIDED2]])
; CHECK-ORDERED: %[[RDX1]] = call float @llvm.vector.reduce.fadd.v4f32(float %[[VEC_PHI2]], <4 x float> %[[STRIDED1]])
; CHECK-ORDERED: for.end
; CHECK-ORDERED: ret void

; CHECK-UNORDERED-LABEL: @fadd_strict_interleave
; CHECK-UNORDERED: %[[ARRAYIDX:.*]] = getelementptr inbounds float, float* %a, i64 1
; CHECK-UNORDERED: %[[LOADA1:.*]] = load float, float* %a
; CHECK-UNORDERED: %[[LOADA2:.*]] = load float, float* %[[ARRAYIDX]]
; CHECK-UNORDERED: vector.ph
; CHECK-UNORDERED: %[[INS2:.*]] = insertelement <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, float %[[LOADA2]], i32 0
; CHECK-UNORDERED: %[[INS1:.*]] = insertelement <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, float %[[LOADA1]], i32 0
; CHECK-UNORDERED: vector.body
; CHECK-UNORDERED: %[[VEC_PHI2:.*]] = phi <4 x float> [ %[[INS2]], %vector.ph ], [ %[[VEC_FADD2:.*]], %vector.body ]
; CHECK-UNORDERED: %[[VEC_PHI1:.*]] = phi <4 x float> [ %[[INS1]], %vector.ph ], [ %[[VEC_FADD1:.*]], %vector.body ]
; CHECK-UNORDERED: %[[WIDE_LOAD:.*]] = load <8 x float>, <8 x float>*
; CHECK-UNORDERED: %[[STRIDED1:.*]] = shufflevector <8 x float> %[[WIDE_LOAD]], <8 x float> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK-UNORDERED: %[[STRIDED2:.*]] = shufflevector <8 x float> %[[WIDE_LOAD]], <8 x float> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
; CHECK-UNORDERED: %[[VEC_FADD1]] = fadd <4 x float> %[[STRIDED1:.*]], %[[VEC_PHI1]]
; CHECK-UNORDERED: %[[VEC_FADD2]] = fadd <4 x float> %[[STRIDED2:.*]], %[[VEC_PHI2]]
; CHECK-UNORDERED-NOT: call float @llvm.vector.reduce.fadd
; CHECK-UNORDERED: middle.block
; CHECK-UNORDERED: %[[RDX1:.*]] = call float @llvm.vector.reduce.fadd.v4f32(float -0.000000e+00, <4 x float> %[[VEC_FADD1]])
; CHECK-UNORDERED: %[[RDX2:.*]] = call float @llvm.vector.reduce.fadd.v4f32(float -0.000000e+00, <4 x float> %[[VEC_FADD2]])
; CHECK-UNORDERED: for.body
; CHECK-UNORDERED: %[[LOAD1:.*]] = load float, float*
; CHECK-UNORDERED: %[[FADD1:.*]] = fadd float %[[LOAD1]], {{.*}}
; CHECK-UNORDERED: %[[LOAD2:.*]] = load float, float*
; CHECK-UNORDERED: %[[FADD2:.*]] = fadd float %[[LOAD2]], {{.*}}
; CHECK-UNORDERED: for.end
; CHECK-UNORDERED: %[[SUM1:.*]] = phi float [ %[[FADD1]], %for.body ], [ %[[RDX1]], %middle.block ]
; CHECK-UNORDERED: %[[SUM2:.*]] = phi float [ %[[FADD2]], %for.body ], [ %[[RDX2]], %middle.block ]
; CHECK-UNORDERED: store float %[[SUM1]]
; CHECK-UNORDERED: store float %[[SUM2]]
; CHECK-UNORDERED: ret void

; CHECK-NOT-VECTORIZED-LABEL: @fadd_strict_interleave
; CHECK-NOT-VECTORIZED-NOT: vector.body

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

define float @fadd_of_sum(float* noalias nocapture readonly %a, float* noalias nocapture readonly %b, i64 %n) {
; CHECK-ORDERED-LABEL: @fadd_of_sum
; CHECK-ORDERED: vector.body
; CHECK-ORDERED: %[[VEC_PHI1:.*]] = phi float [ 0.000000e+00, %vector.ph ], [ %[[RDX:.*]], %vector.body ]
; CHECK-ORDERED: %[[LOAD1:.*]] = load <4 x float>, <4 x float>*
; CHECK-ORDERED: %[[LOAD2:.*]] = load <4 x float>, <4 x float>*
; CHECK-ORDERED: %[[ADD:.*]] = fadd <4 x float> %[[LOAD1]], %[[LOAD2]]
; CHECK-ORDERED: %[[RDX]] = call float @llvm.vector.reduce.fadd.v4f32(float %[[VEC_PHI1]], <4 x float> %[[ADD]])
; CHECK-ORDERED: for.end.loopexit
; CHECK-ORDERED: %[[EXIT_PHI:.*]] = phi float [ %[[SCALAR:.*]], %for.body ], [ %[[RDX]], %middle.block ]
; CHECK-ORDERED: for.end
; CHECK-ORDERED: %[[PHI:.*]] = phi float [ 0.000000e+00, %entry ], [ %[[EXIT_PHI]], %for.end.loopexit ]
; CHECK-ORDERED: ret float %[[PHI]]

; CHECK-UNORDERED-LABEL: @fadd_of_sum
; CHECK-UNORDERED: vector.body
; CHECK-UNORDERED: %[[VEC_PHI:.*]] = phi <4 x float> [ <float 0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %vector.ph ], [ %[[VEC_FADD2:.*]], %vector.body ]
; CHECK-UNORDERED: %[[VEC_LOAD1:.*]] = load <4 x float>, <4 x float>*
; CHECK-UNORDERED: %[[VEC_LOAD2:.*]] = load <4 x float>, <4 x float>*
; CHECK-UNORDERED: %[[VEC_FADD1:.*]] = fadd <4 x float> %[[VEC_LOAD1]], %[[VEC_LOAD2]]
; CHECK-UNORDERED: %[[VEC_FADD2]] = fadd <4 x float> %[[VEC_PHI]], %[[VEC_FADD1]]
; CHECK-UNORDERED-NOT: call float @llvm.vector.reduce.fadd
; CHECK-UNORDERED: middle.block
; CHECK-UNORDERED: %[[RDX:.*]] = call float @llvm.vector.reduce.fadd.v4f32(float -0.000000e+00, <4 x float> %[[VEC_FADD2]])
; CHECK-UNORDERED: for.body
; CHECK-UNORDERED: %[[LOAD1:.*]] = load float, float*
; CHECK-UNORDERED: %[[LOAD2:.*]] = load float, float*
; CHECK-UNORDERED: %[[FADD1:.*]] = fadd float %[[LOAD1]], %[[LOAD2]]
; CHECK-UNORDERED: %[[FADD2:.*]] = fadd float {{.*}}, %[[FADD1]]
; CHECK-UNORDERED: for.end.loopexit
; CHECK-UNORDERED: %[[EXIT:.*]] = phi float [ %[[FADD2]], %for.body ], [ %[[RDX]], %middle.block ]
; CHECK-UNORDERED: for.end
; CHECK-UNORDERED: %[[SUM:.*]] = phi float [ 0.000000e+00, %entry ], [ %[[EXIT]], %for.end.loopexit ]
; CHECK-UNORDERED: ret float %[[SUM]]

; CHECK-NOT-VECTORIZED-LABEL: @fadd_of_sum
; CHECK-NOT-VECTORIZED-NOT: vector.body

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
; CHECK-ORDERED-LABEL: @fadd_conditional
; CHECK-ORDERED: vector.body:
; CHECK-ORDERED: %[[PHI:.*]] = phi float [ 1.000000e+00, %vector.ph ], [ %[[RDX:.*]], %pred.load.continue6 ]
; CHECK-ORDERED: %[[LOAD1:.*]] = load <4 x float>, <4 x float>*
; CHECK-ORDERED: %[[FCMP1:.*]] = fcmp une <4 x float> %[[LOAD1]], zeroinitializer
; CHECK-ORDERED: %[[EXTRACT:.*]] = extractelement <4 x i1> %[[FCMP1]], i32 0
; CHECK-ORDERED: br i1 %[[EXTRACT]], label %pred.load.if, label %pred.load.continue
; CHECK-ORDERED: pred.load.continue6
; CHECK-ORDERED: %[[PHI1:.*]] = phi <4 x float> [ %[[PHI0:.*]], %pred.load.continue4 ], [ %[[INS_ELT:.*]], %pred.load.if5 ]
; CHECK-ORDERED: %[[XOR:.*]] =  xor <4 x i1> %[[FCMP1]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-ORDERED: %[[PRED:.*]] = select <4 x i1> %[[XOR]], <4 x float> <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>, <4 x float> %[[PHI1]]
; CHECK-ORDERED: %[[RDX]] = call float @llvm.vector.reduce.fadd.v4f32(float %[[PHI]], <4 x float> %[[PRED]])
; CHECK-ORDERED: for.body
; CHECK-ORDERED: %[[RES_PHI:.*]] = phi float [ %[[MERGE_RDX:.*]], %scalar.ph ], [ %[[FADD:.*]], %for.inc ]
; CHECK-ORDERED: %[[LOAD2:.*]] = load float, float*
; CHECK-ORDERED: %[[FCMP2:.*]] = fcmp une float %[[LOAD2]], 0.000000e+00
; CHECK-ORDERED: br i1 %[[FCMP2]], label %if.then, label %for.inc
; CHECK-ORDERED: if.then
; CHECK-ORDERED: %[[LOAD3:.*]] = load float, float*
; CHECK-ORDERED: br label %for.inc
; CHECK-ORDERED: for.inc
; CHECK-ORDERED: %[[PHI2:.*]] = phi float [ %[[LOAD3]], %if.then ], [ 3.000000e+00, %for.body ]
; CHECK-ORDERED: %[[FADD]] = fadd float %[[RES_PHI]], %[[PHI2]]
; CHECK-ORDERED: for.end
; CHECK-ORDERED: %[[RDX_PHI:.*]] = phi float [ %[[FADD]], %for.inc ], [ %[[RDX]], %middle.block ]
; CHECK-ORDERED: ret float %[[RDX_PHI]]

; CHECK-UNORDERED-LABEL: @fadd_conditional
; CHECK-UNORDERED: vector.body
; CHECK-UNORDERED: %[[PHI:.*]] = phi <4 x float> [ <float 1.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %vector.ph ], [ %[[VEC_FADD:.*]], %pred.load.continue6 ]
; CHECK-UNORDERED: %[[LOAD1:.*]] = load <4 x float>, <4 x float>*
; CHECK-UNORDERED: %[[FCMP1:.*]] = fcmp une <4 x float> %[[LOAD1]], zeroinitializer
; CHECK-UNORDERED: %[[EXTRACT:.*]] = extractelement <4 x i1> %[[FCMP1]], i32 0
; CHECK-UNORDERED: br i1 %[[EXTRACT]], label %pred.load.if, label %pred.load.continue
; CHECK-UNORDERED: pred.load.continue6
; CHECK-UNORDERED: %[[XOR:.*]] =  xor <4 x i1> %[[FCMP1]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-UNORDERED: %[[PRED:.*]] = select <4 x i1> %[[XOR]], <4 x float> <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>, <4 x float> %[[PRED_PHI:.*]]
; CHECK-UNORDERED: %[[VEC_FADD]] = fadd <4 x float> %[[PHI]], %[[PRED]]
; CHECK-UNORDERED-NOT: call float @llvm.vector.reduce.fadd
; CHECK-UNORDERED: middle.block
; CHECK-UNORDERED: %[[RDX:.*]] = call float @llvm.vector.reduce.fadd.v4f32(float -0.000000e+00, <4 x float> %[[VEC_FADD]])
; CHECK-UNORDERED: for.body
; CHECK-UNORDERED: %[[RES_PHI:.*]] = phi float [ %[[MERGE_RDX:.*]], %scalar.ph ], [ %[[FADD:.*]], %for.inc ]
; CHECK-UNORDERED: %[[LOAD2:.*]] = load float, float*
; CHECK-UNORDERED: %[[FCMP2:.*]] = fcmp une float %[[LOAD2]], 0.000000e+00
; CHECK-UNORDERED: br i1 %[[FCMP2]], label %if.then, label %for.inc
; CHECK-UNORDERED: if.then
; CHECK-UNORDERED: %[[LOAD3:.*]] = load float, float*
; CHECK-UNORDERED: for.inc
; CHECK-UNORDERED: %[[PHI:.*]] = phi float [ %[[LOAD3]], %if.then ], [ 3.000000e+00, %for.body ]
; CHECK-UNORDERED: %[[FADD]] = fadd float %[[RES_PHI]], %[[PHI]]
; CHECK-UNORDERED: for.end
; CHECK-UNORDERED: %[[RDX_PHI:.*]] = phi float [ %[[FADD]], %for.inc ], [ %[[RDX]], %middle.block ]
; CHECK-UNORDERED: ret float %[[RDX_PHI]]

; CHECK-NOT-VECTORIZED-LABEL: @fadd_conditional
; CHECK-NOT-VECTORIZED-NOT: vector.body

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
; CHECK-ORDERED-LABEL: @fadd_predicated
; CHECK-ORDERED: vector.ph
; CHECK-ORDERED: %[[TRIP_MINUS_ONE:.*]] = sub i64 %n, 1
; CHECK-ORDERED: %[[BROADCAST_INS:.*]] = insertelement <2 x i64> poison, i64 %[[TRIP_MINUS_ONE]], i32 0
; CHECK-ORDERED: %[[SPLAT:.*]] = shufflevector <2 x i64> %[[BROADCAST_INS]], <2 x i64> poison, <2 x i32> zeroinitializer
; CHECK-ORDERED: vector.body
; CHECK-ORDERED: %[[RDX_PHI:.*]] =  phi float [ 0.000000e+00, %vector.ph ], [ %[[RDX:.*]], %pred.load.continue2 ]
; CHECK-ORDERED: pred.load.continue2
; CHECK-ORDERED: %[[PHI:.*]] = phi <2 x float> [ %[[PHI0:.*]], %pred.load.continue ], [ %[[INS_ELT:.*]], %pred.load.if1 ]
; CHECK-ORDERED: %[[MASK:.*]] = select <2 x i1> %0, <2 x float> %[[PHI]], <2 x float> <float -0.000000e+00, float -0.000000e+00>
; CHECK-ORDERED: %[[RDX]] = call float @llvm.vector.reduce.fadd.v2f32(float %[[RDX_PHI]], <2 x float> %[[MASK]])
; CHECK-ORDERED: for.end:
; CHECK-ORDERED: %[[RES_PHI:.*]] = phi float [ %[[FADD:.*]], %for.body ], [ %[[RDX]], %middle.block ]
; CHECK-ORDERED: ret float %[[RES_PHI]]

; CHECK-UNORDERED-LABEL: @fadd_predicated
; CHECK-UNORDERED: vector.ph
; CHECK-UNORDERED: %[[TRIP_MINUS_ONE:.*]] = sub i64 %n, 1
; CHECK-UNORDERED: %[[BROADCAST_INS:.*]] = insertelement <2 x i64> poison, i64 %[[TRIP_MINUS_ONE]], i32 0
; CHECK-UNORDERED: %[[SPLAT:.*]] = shufflevector <2 x i64> %[[BROADCAST_INS]], <2 x i64> poison, <2 x i32> zeroinitializer
; CHECK-UNORDERED: vector.body
; CHECK-UNORDERED: %[[RDX_PHI:.*]] =  phi <2 x float> [ <float 0.000000e+00, float -0.000000e+00>, %vector.ph ], [ %[[FADD:.*]], %pred.load.continue2 ]
; CHECK-UNORDERED: %[[ICMP:.*]] = icmp ule <2 x i64> %vec.ind, %[[SPLAT]]
; CHECK-UNORDERED: pred.load.continue2
; CHECK-UNORDERED: %[[FADD]] = fadd <2 x float> %[[RDX_PHI]], {{.*}}
; CHECK-UNORDERED: %[[MASK:.*]] = select <2 x i1> %[[ICMP]], <2 x float> %[[FADD]], <2 x float> %[[RDX_PHI]]
; CHECK-UNORDERED-NOT: call float @llvm.vector.reduce.fadd
; CHECK-UNORDERED: middle.block
; CHECK-UNORDERED: %[[RDX:.*]] = call float @llvm.vector.reduce.fadd.v2f32(float -0.000000e+00, <2 x float> %[[MASK]])
; CHECK-UNORDERED: for.body
; CHECK-UNORDERED: %[[LOAD:.*]] = load float, float*
; CHECK-UNORDERED: %[[FADD2:.*]] = fadd float {{.*}}, %[[LOAD]]
; CHECK-UNORDERED: for.end
; CHECK-UNORDERED: %[[SUM:.*]] = phi float [ %[[FADD2]], %for.body ], [ %[[RDX]], %middle.block ]
; CHECK-UNORDERED: ret float %[[SUM]]

; CHECK-NOT-VECTORIZED-LABEL: @fadd_predicated
; CHECK-NOT-VECTORIZED-NOT: vector.body

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
; CHECK-ORDERED-LABEL: @fadd_multiple
; CHECK-ORDERED-NOT: vector.body

; CHECK-UNORDERED-LABEL: @fadd_multiple
; CHECK-UNORDERED: vector.body
; CHECK-UNORDERED: %[[PHI:.*]] = phi <8 x float> [ <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %vector.ph ], [ %[[VEC_FADD2:.*]], %vector.body ]
; CHECK-UNORDERED: %[[VEC_LOAD1:.*]] = load <8 x float>, <8 x float>
; CHECK-UNORDERED: %[[VEC_FADD1:.*]] = fadd <8 x float> %[[PHI]], %[[VEC_LOAD1]]
; CHECK-UNORDERED: %[[VEC_LOAD2:.*]] = load <8 x float>, <8 x float>
; CHECK-UNORDERED: %[[VEC_FADD2]] = fadd <8 x float> %[[VEC_FADD1]], %[[VEC_LOAD2]]
; CHECK-UNORDERED: middle.block
; CHECK-UNORDERED: %[[RDX:.*]] = call float @llvm.vector.reduce.fadd.v8f32(float -0.000000e+00, <8 x float> %[[VEC_FADD2]])
; CHECK-UNORDERED: for.body
; CHECK-UNORDERED: %[[SUM:.*]] = phi float [ %bc.merge.rdx, %scalar.ph ], [ %[[FADD2:.*]], %for.body ]
; CHECK-UNORDERED: %[[LOAD1:.*]] = load float, float*
; CHECK-UNORDERED: %[[FADD1:.*]] = fadd float %sum, %[[LOAD1]]
; CHECK-UNORDERED: %[[LOAD2:.*]] = load float, float*
; CHECK-UNORDERED: %[[FADD2]] = fadd float %[[FADD1]], %[[LOAD2]]
; CHECK-UNORDERED: for.end
; CHECK-UNORDERED: %[[RET:.*]] = phi float [ %[[FADD2]], %for.body ], [ %[[RDX]], %middle.block ]
; CHECK-UNORDERED: ret float %[[RET]]

; CHECK-NOT-VECTORIZED-LABEL: @fadd_multiple
; CHECK-NOT-VECTORIZED-NOT: vector.body

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

; Negative test - loop contains two fadds and only one fadd has the fast flag,
; which we cannot safely reorder.
define float @fadd_multiple_one_flag(float* noalias nocapture %a, float* noalias nocapture %b, i64 %n) {
; CHECK-ORDERED-LABEL: @fadd_multiple_one_flag
; CHECK-ORDERED-NOT: vector.body

; CHECK-UNORDERED-LABEL: @fadd_multiple_one_flag
; CHECK-UNORDERED: vector.body
; CHECK-UNORDERED: %[[PHI:.*]] = phi <8 x float> [ <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %vector.ph ], [ %[[VEC_FADD2:.*]], %vector.body ]
; CHECK-UNORDERED: %[[VEC_LOAD1:.*]] = load <8 x float>, <8 x float>
; CHECK-UNORDERED: %[[VEC_FADD1:.*]] = fadd <8 x float> %[[PHI]], %[[VEC_LOAD1]]
; CHECK-UNORDERED: %[[VEC_LOAD2:.*]] = load <8 x float>, <8 x float>
; CHECK-UNORDERED: %[[VEC_FADD2]] = fadd fast <8 x float> %[[VEC_FADD1]], %[[VEC_LOAD2]]
; CHECK-UNORDERED: middle.block
; CHECK-UNORDERED: %[[RDX:.*]] = call float @llvm.vector.reduce.fadd.v8f32(float -0.000000e+00, <8 x float> %[[VEC_FADD2]])
; CHECK-UNORDERED: for.body
; CHECK-UNORDERED: %[[SUM:.*]] = phi float [ %bc.merge.rdx, %scalar.ph ], [ %[[FADD2:.*]], %for.body ]
; CHECK-UNORDERED: %[[LOAD1:.*]] = load float, float*
; CHECK-UNORDERED: %[[FADD1:.*]] = fadd float %sum, %[[LOAD1]]
; CHECK-UNORDERED: %[[LOAD2:.*]] = load float, float*
; CHECK-UNORDERED: %[[FADD2]] = fadd fast float %[[FADD1]], %[[LOAD2]]
; CHECK-UNORDERED: for.end
; CHECK-UNORDERED: %[[RET:.*]] = phi float [ %[[FADD2]], %for.body ], [ %[[RDX]], %middle.block ]
; CHECK-UNORDERED: ret float %[[RET]]

; CHECK-NOT-VECTORIZED-LABEL: @fadd_multiple_one_flag
; CHECK-NOT-VECTORIZED-NOT: vector.body

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
  %add3 = fadd fast float %add, %1
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:                                         ; preds = %for.body
  %rdx = phi float [ %add3, %for.body ]
  ret float %rdx
}

; Tests with both a floating point reduction & induction, e.g.
;
;float fp_iv_rdx_loop(float *values, float init, float * __restrict__ A, int N) {
;  float fp_inc = 2.0;
;  float x = init;
;  float sum = 0.0;
;  for (int i=0; i < N; ++i) {
;    A[i] = x;
;    x += fp_inc;
;    sum += values[i];
;  }
;  return sum;
;}
;

; Strict reduction could be performed in-loop, but ordered FP induction variables are not supported
; Note: This test does not use metadata hints, and as such we should not expect the CHECK-UNORDERED case to vectorize, even
; with the -hints-allow-reordering flag set to true.
define float @induction_and_reduction(float* nocapture readonly %values, float %init, float* noalias nocapture %A, i64 %N) {
; CHECK-ORDERED-LABEL: @induction_and_reduction
; CHECK-ORDERED-NOT: vector.body

; CHECK-UNORDERED-LABEL: @induction_and_reduction
; CHECK-UNORDERED-NOT: vector.body

; CHECK-NOT-VECTORIZED-LABEL: @induction_and_reduction
; CHECK-NOT-VECTORIZED-NOT: vector.body

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.015 = phi float [ 0.000000e+00, %entry ], [ %add3, %for.body ]
  %x.014 = phi float [ %init, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %A, i64 %iv
  store float %x.014, float* %arrayidx, align 4
  %add = fadd float %x.014, 2.000000e+00
  %arrayidx2 = getelementptr inbounds float, float* %values, i64 %iv
  %0 = load float, float* %arrayidx2, align 4
  %add3 = fadd float %sum.015, %0
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  ret float %add3
}

; As above, but with the FP induction being unordered (fast) the loop can be vectorized with strict reductions
define float @fast_induction_and_reduction(float* nocapture readonly %values, float %init, float* noalias nocapture %A, i64 %N) {
; CHECK-ORDERED-LABEL: @fast_induction_and_reduction
; CHECK-ORDERED: vector.ph
; CHECK-ORDERED: %[[INDUCTION:.*]] = fadd fast <4 x float> {{.*}}, <float 0.000000e+00, float 2.000000e+00, float 4.000000e+00, float 6.000000e+00>
; CHECK-ORDERED: vector.body
; CHECK-ORDERED: %[[RDX_PHI:.*]] = phi float [ 0.000000e+00, %vector.ph ], [ %[[FADD2:.*]], %vector.body ]
; CHECK-ORDERED: %[[IND_PHI:.*]] = phi <4 x float> [ %[[INDUCTION]], %vector.ph ], [ %[[VEC_IND_NEXT:.*]], %vector.body ]
; CHECK-ORDERED: %[[LOAD1:.*]] = load <4 x float>, <4 x float>*
; CHECK-ORDERED: %[[FADD1:.*]] = call float @llvm.vector.reduce.fadd.v4f32(float %[[RDX_PHI]], <4 x float> %[[LOAD1]])
; CHECK-ORDERED: %[[VEC_IND_NEXT]] = fadd fast <4 x float> %[[IND_PHI]], <float 8.000000e+00, float 8.000000e+00, float 8.000000e+00, float 8.000000e+00>
; CHECK-ORDERED: for.body
; CHECK-ORDERED: %[[RDX_SUM_PHI:.*]] = phi float [ {{.*}}, %scalar.ph ], [ %[[FADD2:.*]], %for.body ]
; CHECK-ORDERED: %[[IND_SUM_PHI:.*]] = phi fast float [ {{.*}}, %scalar.ph ], [ %[[ADD_IND:.*]], %for.body ]
; CHECK-ORDERED: store float %[[IND_SUM_PHI]], float*
; CHECK-ORDERED: %[[ADD_IND]] = fadd fast float %[[IND_SUM_PHI]], 2.000000e+00
; CHECK-ORDERED: %[[LOAD2:.*]] = load float, float*
; CHECK-ORDERED: %[[FADD2]] = fadd float %[[RDX_SUM_PHI]], %[[LOAD2]]
; CHECK-ORDERED: for.end
; CHECK-ORDERED: %[[RES_PHI:.*]] = phi float [ %[[FADD2]], %for.body ], [ %[[FADD1]], %middle.block ]
; CHECK-ORDERED: ret float %[[RES_PHI]]

; CHECK-UNORDERED-LABEL: @fast_induction_and_reduction
; CHECK-UNORDERED: vector.ph
; CHECK-UNORDERED: %[[INDUCTION:.*]] = fadd fast <4 x float> {{.*}}, <float 0.000000e+00, float 2.000000e+00, float 4.000000e+00, float 6.000000e+00>
; CHECK-UNORDERED: vector.body
; CHECK-UNORDERED: %[[RDX_PHI:.*]] = phi <4 x float> [ <float 0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %vector.ph ], [ %[[VEC_FADD:.*]], %vector.body ]
; CHECK-UNORDERED: %[[IND_PHI:.*]] = phi <4 x float> [ %[[INDUCTION]], %vector.ph ], [ %[[VEC_IND_NEXT:.*]], %vector.body ]
; CHECK-UNORDERED: %[[LOAD1:.*]] = load <4 x float>, <4 x float>*
; CHECK-UNORDERED: %[[VEC_FADD]] = fadd <4 x float> %[[RDX_PHI]], %[[LOAD1]]
; CHECK-UNORDERED: %[[VEC_IND_NEXT]] = fadd fast <4 x float> %[[IND_PHI]], <float 8.000000e+00, float 8.000000e+00, float 8.000000e+00, float 8.000000e+00>
; CHECK-UNORDERED: middle.block:
; CHECK-UNORDERED: %[[VEC_RDX:.*]] = call float @llvm.vector.reduce.fadd.v4f32(float -0.000000e+00, <4 x float> %[[VEC_FADD]])
; CHECK-UNORDERED: for.body:
; CHECK-UNORDERED: %[[RDX_SUM_PHI:.*]] = phi float [ {{.*}}, %scalar.ph ], [ %[[FADD:.*]], %for.body ]
; CHECK-UNORDERED: %[[IND_SUM_PHI:.*]] = phi fast float [ {{.*}}, %scalar.ph ], [ %[[ADD_IND:.*]], %for.body ]
; CHECK-UNORDERED: store float %[[IND_SUM_PHI]], float*
; CHECK-UNORDERED: %[[ADD_IND]] = fadd fast float %[[IND_SUM_PHI]], 2.000000e+00
; CHECK-UNORDERED: %[[LOAD2:.*]] = load float, float*
; CHECK-UNORDERED: %[[FADD]] = fadd float %[[RDX_SUM_PHI]], %[[LOAD2]]
; CHECK-UNORDERED: for.end
; CHECK-UNORDERED: %[[RES_PHI:.*]] = phi float [ %[[FADD]], %for.body ], [ %[[VEC_RDX]], %middle.block ]
; CHECK-UNORDERED: ret float %[[RES_PHI]]

; CHECK-NOT-VECTORIZED-LABEL: @fast_induction_and_reduction
; CHECK-NOT-VECTORIZED-NOT: vector.body

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.015 = phi float [ 0.000000e+00, %entry ], [ %add3, %for.body ]
  %x.014 = phi fast float [ %init, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %A, i64 %iv
  store float %x.014, float* %arrayidx, align 4
  %add = fadd fast float %x.014, 2.000000e+00
  %arrayidx2 = getelementptr inbounds float, float* %values, i64 %iv
  %0 = load float, float* %arrayidx2, align 4
  %add3 = fadd float %sum.015, %0
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !2

for.end:
  ret float %add3
}

; The FP induction is fast, but here we can't vectorize as only one of the reductions is an FAdd that can be performed in-loop
; Note: This test does not use metadata hints, and as such we should not expect the CHECK-UNORDERED case to vectorize, even
; with the -hints-allow-reordering flag set to true.
define float @fast_induction_unordered_reduction(float* nocapture readonly %values, float %init, float* noalias nocapture %A, float* noalias nocapture %B, i64 %N) {

; CHECK-ORDERED-LABEL: @fast_induction_unordered_reduction
; CHECK-ORDERED-NOT: vector.body

; CHECK-UNORDERED-LABEL: @fast_induction_unordered_reduction
; CHECK-UNORDERED-NOT: vector.body

; CHECK-NOT-VECTORIZED-LABEL: @fast_induction_unordered_reduction
; CHECK-NOT-VECTORIZED-NOT: vector.body

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum2.023 = phi float [ 3.000000e+00, %entry ], [ %mul, %for.body ]
  %sum.022 = phi float [ 0.000000e+00, %entry ], [ %add3, %for.body ]
  %x.021 = phi float [ %init, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %A, i64 %iv
  store float %x.021, float* %arrayidx, align 4
  %add = fadd fast float %x.021, 2.000000e+00
  %arrayidx2 = getelementptr inbounds float, float* %values, i64 %iv
  %0 = load float, float* %arrayidx2, align 4
  %add3 = fadd float %sum.022, %0
  %mul = fmul float %sum2.023, %0
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  %add6 = fadd float %add3, %mul
  ret float %add6
}

; Test reductions for a VF of 1 and a UF > 1.
define float @fadd_scalar_vf(float* noalias nocapture readonly %a, i64 %n) {
; CHECK-ORDERED-LABEL: @fadd_scalar_vf
; CHECK-ORDERED: vector.body
; CHECK-ORDERED: %[[VEC_PHI:.*]] = phi float [ 0.000000e+00, {{.*}} ], [ %[[FADD4:.*]], %vector.body ]
; CHECK-ORDERED: %[[LOAD1:.*]] = load float, float*
; CHECK-ORDERED: %[[LOAD2:.*]] = load float, float*
; CHECK-ORDERED: %[[LOAD3:.*]] = load float, float*
; CHECK-ORDERED: %[[LOAD4:.*]] = load float, float*
; CHECK-ORDERED: %[[FADD1:.*]] = fadd float %[[VEC_PHI]], %[[LOAD1]]
; CHECK-ORDERED: %[[FADD2:.*]] = fadd float %[[FADD1]], %[[LOAD2]]
; CHECK-ORDERED: %[[FADD3:.*]] = fadd float %[[FADD2]], %[[LOAD3]]
; CHECK-ORDERED: %[[FADD4]] = fadd float %[[FADD3]], %[[LOAD4]]
; CHECK-ORDERED-NOT: call float @llvm.vector.reduce.fadd
; CHECK-ORDERED: scalar.ph
; CHECK-ORDERED: %[[MERGE_RDX:.*]] = phi float [ 0.000000e+00, %entry ], [ %[[FADD4]], %middle.block ]
; CHECK-ORDERED: for.body
; CHECK-ORDERED: %[[SUM_PHI:.*]] = phi float [ %[[MERGE_RDX]], %scalar.ph ], [ %[[FADD5:.*]], %for.body ]
; CHECK-ORDERED: %[[LOAD5:.*]] = load float, float*
; CHECK-ORDERED: %[[FADD5]] = fadd float %[[LOAD5]], %[[SUM_PHI]]
; CHECK-ORDERED: for.end
; CHECK-ORDERED: %[[RES_PHI:.*]] = phi float [ %[[FADD5]], %for.body ], [ %[[FADD4]], %middle.block ]
; CHECK-ORDERED: ret float %[[RES_PHI]]

; CHECK-UNORDERED-LABEL: @fadd_scalar_vf
; CHECK-UNORDERED: vector.body
; CHECK-UNORDERED: %[[VEC_PHI1:.*]] = phi float [ 0.000000e+00, %vector.ph ], [ %[[FADD1:.*]], %vector.body ]
; CHECK-UNORDERED: %[[VEC_PHI2:.*]] = phi float [ -0.000000e+00, %vector.ph ], [ %[[FADD2:.*]], %vector.body ]
; CHECK-UNORDERED: %[[VEC_PHI3:.*]] = phi float [ -0.000000e+00, %vector.ph ], [ %[[FADD3:.*]], %vector.body ]
; CHECK-UNORDERED: %[[VEC_PHI4:.*]] = phi float [ -0.000000e+00, %vector.ph ], [ %[[FADD4:.*]], %vector.body ]
; CHECK-UNORDERED: %[[LOAD1:.*]] = load float, float*
; CHECK-UNORDERED: %[[LOAD2:.*]] = load float, float*
; CHECK-UNORDERED: %[[LOAD3:.*]] = load float, float*
; CHECK-UNORDERED: %[[LOAD4:.*]] = load float, float*
; CHECK-UNORDERED: %[[FADD1]] = fadd float %[[LOAD1]], %[[VEC_PHI1]]
; CHECK-UNORDERED: %[[FADD2]] = fadd float %[[LOAD2]], %[[VEC_PHI2]]
; CHECK-UNORDERED: %[[FADD3]] = fadd float %[[LOAD3]], %[[VEC_PHI3]]
; CHECK-UNORDERED: %[[FADD4]] = fadd float %[[LOAD4]], %[[VEC_PHI4]]
; CHECK-UNORDERED-NOT: call float @llvm.vector.reduce.fadd
; CHECK-UNORDERED: middle.block
; CHECK-UNORDERED: %[[BIN_RDX1:.*]] = fadd float %[[FADD2]], %[[FADD1]]
; CHECK-UNORDERED: %[[BIN_RDX2:.*]] = fadd float %[[FADD3]], %[[BIN_RDX1]]
; CHECK-UNORDERED: %[[BIN_RDX3:.*]] = fadd float %[[FADD4]], %[[BIN_RDX2]]
; CHECK-UNORDERED: scalar.ph
; CHECK-UNORDERED: %[[MERGE_RDX:.*]] = phi float [ 0.000000e+00, %entry ], [ %[[BIN_RDX3]], %middle.block ]
; CHECK-UNORDERED: for.body
; CHECK-UNORDERED: %[[SUM_PHI:.*]] = phi float [ %[[MERGE_RDX]], %scalar.ph ], [ %[[FADD5:.*]], %for.body ]
; CHECK-UNORDERED: %[[LOAD5:.*]] = load float, float*
; CHECK-UNORDERED: %[[FADD5]] = fadd float %[[LOAD5]], %[[SUM_PHI]]
; CHECK-UNORDERED: for.end
; CHECK-UNORDERED: %[[RES_PHI:.*]] = phi float [ %[[FADD5]], %for.body ], [ %[[BIN_RDX3]], %middle.block ]
; CHECK-UNORDERED: ret float %[[RES_PHI]]

; CHECK-NOT-VECTORIZED-LABEL: @fadd_scalar_vf
; CHECK-NOT-VECTORIZED-NOT: @vector.body

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
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !4

for.end:
  ret float %add
}

; Same as above but where fadd has a fast-math flag.
define float @fadd_scalar_vf_fmf(float* noalias nocapture readonly %a, i64 %n) {
; CHECK-ORDERED-LABEL: @fadd_scalar_vf_fmf
; CHECK-ORDERED: vector.body:
; CHECK-ORDERED: [[VEC_PHI:%.*]] = phi float [ 0.000000e+00, %vector.ph ], [ [[FADD4:%.*]], %vector.body ]
; CHECK-ORDERED: [[LOAD1:%.*]] = load float, float*
; CHECK-ORDERED: [[LOAD2:%.*]] = load float, float*
; CHECK-ORDERED: [[LOAD3:%.*]] = load float, float*
; CHECK-ORDERED: [[LOAD4:%.*]] = load float, float*
; CHECK-ORDERED: [[FADD1:%.*]] = fadd nnan float [[VEC_PHI]], [[LOAD1]]
; CHECK-ORDERED: [[FADD2:%.*]] = fadd nnan float [[FADD1]], [[LOAD2]]
; CHECK-ORDERED: [[FADD3:%.*]] = fadd nnan float [[FADD2]], [[LOAD3]]
; CHECK-ORDERED: [[FADD4]] = fadd nnan float [[FADD3]], [[LOAD4]]
; CHECK-ORDERED-NOT: @llvm.vector.reduce.fadd
; CHECK-ORDERED: scalar.ph:
; CHECK-ORDERED: [[MERGE_RDX:%.*]] = phi float [ 0.000000e+00, %entry ], [ [[FADD4]], %middle.block ]
; CHECK-ORDERED: for.body:
; CHECK-ORDERED: [[SUM_07:%.*]] = phi float [ [[MERGE_RDX]], %scalar.ph ], [ [[FADD5:%.*]], %for.body ]
; CHECK-ORDERED: [[LOAD5:%.*]] = load float, float*
; CHECK-ORDERED: [[FADD5]] = fadd nnan float [[LOAD5]], [[SUM_07]]
; CHECK-ORDERED: for.end:
; CHECK-ORDERED: [[RES:%.*]] = phi float [ [[FADD5]], %for.body ], [ [[FADD4]], %middle.block ]
; CHECK-ORDERED: ret float [[RES]]

; CHECK-UNORDERED-LABEL: @fadd_scalar_vf_fmf
; CHECK-UNORDERED: vector.body:
; CHECK-UNORDERED: [[VEC_PHI1:%.*]] = phi float [ 0.000000e+00, %vector.ph ], [ [[FADD1:%.*]], %vector.body ]
; CHECK-UNORDERED: [[VEC_PHI2:%.*]] = phi float [ -0.000000e+00, %vector.ph ], [ [[FADD2:%.*]], %vector.body ]
; CHECK-UNORDERED: [[VEC_PHI3:%.*]] = phi float [ -0.000000e+00, %vector.ph ], [ [[FADD3:%.*]], %vector.body ]
; CHECK-UNORDERED: [[VEC_PHI4:%.*]] = phi float [ -0.000000e+00, %vector.ph ], [ [[FADD4:%.*]], %vector.body ]
; CHECK-UNORDERED: [[LOAD1:%.*]] = load float, float*
; CHECK-UNORDERED: [[LOAD2:%.*]] = load float, float*
; CHECK-UNORDERED: [[LOAD3:%.*]] = load float, float*
; CHECK-UNORDERED: [[LOAD4:%.*]] = load float, float*
; CHECK-UNORDERED: [[FADD1]] = fadd nnan float [[LOAD1]], [[VEC_PHI1]]
; CHECK-UNORDERED: [[FADD2]] = fadd nnan float [[LOAD2]], [[VEC_PHI2]]
; CHECK-UNORDERED: [[FADD3]] = fadd nnan float [[LOAD3]], [[VEC_PHI3]]
; CHECK-UNORDERED: [[FADD4]] = fadd nnan float [[LOAD4]], [[VEC_PHI4]]
; CHECK-UNORDERED-NOT: @llvm.vector.reduce.fadd
; CHECK-UNORDERED: middle.block:
; CHECK-UNORDERED: [[BIN_RDX1:%.*]] = fadd nnan float [[FADD2]], [[FADD1]]
; CHECK-UNORDERED: [[BIN_RDX2:%.*]] = fadd nnan float [[FADD3]], [[BIN_RDX1]]
; CHECK-UNORDERED: [[BIN_RDX3:%.*]] = fadd nnan float [[FADD4]], [[BIN_RDX2]]
; CHECK-UNORDERED: scalar.ph:
; CHECK-UNORDERED: [[MERGE_RDX:%.*]] = phi float [ 0.000000e+00, %entry ], [ [[BIN_RDX3]], %middle.block ]
; CHECK-UNORDERED: for.body:
; CHECK-UNORDERED: [[SUM_07:%.*]] = phi float [ [[MERGE_RDX]], %scalar.ph ], [ [[FADD5:%.*]], %for.body ]
; CHECK-UNORDERED: [[LOAD5:%.*]] = load float, float*
; CHECK-UNORDERED: [[FADD5]] = fadd nnan float [[LOAD5]], [[SUM_07]]
; CHECK-UORDERED: for.end
; CHECK-UNORDERED: [[RES:%.*]] = phi float [ [[FADD5]], %for.body ], [ [[BIN_RDX3]], %middle.block ]
; CHECK-UNORDERED: ret float [[RES]]

; CHECK-NOT-VECTORIZED-LABEL: @fadd_scalar_vf_fmf
; CHECK-NOT-VECTORIZED-NOT: vector.body

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi float [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %a, i64 %iv
  %0 = load float, float* %arrayidx, align 4
  %add = fadd nnan float %0, %sum.07
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !4

for.end:
  ret float %add
}

; Test case where the reduction step is a first-order recurrence.
define double @reduction_increment_by_first_order_recurrence() {
; CHECK-ORDERED-LABEL: @reduction_increment_by_first_order_recurrence(
; CHECK-ORDERED:  vector.body:
; CHECK-ORDERED:    [[RED:%.*]] = phi double [ 0.000000e+00, %vector.ph ], [ [[RED_NEXT:%.*]], %vector.body ]
; CHECK-ORDERED:    [[VECTOR_RECUR:%.*]] = phi <4 x double> [ <double poison, double poison, double poison, double 0.000000e+00>, %vector.ph ], [ [[FOR_NEXT:%.*]], %vector.body ]
; CHECK-ORDERED:    [[FOR_NEXT]] = sitofp <4 x i32> %vec.ind to <4 x double>
; CHECK-ORDERED:    [[TMP1:%.*]] = shufflevector <4 x double> [[VECTOR_RECUR]], <4 x double> [[FOR_NEXT]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-ORDERED:    [[RED_NEXT]] = call double @llvm.vector.reduce.fadd.v4f64(double [[RED]], <4 x double> [[TMP1]])
; CHECK-ORDERED:  scalar.ph:
; CHECK-ORDERED:    = phi double [ 0.000000e+00, %entry ], [ [[RED_NEXT]], %middle.block ]
;
; CHECK-UNORDERED-LABEL: @reduction_increment_by_first_order_recurrence(
; CHECK-UNORDERED:  vector.body:
; CHECK-UNORDERED:    [[RED:%.*]] = phi <4 x double> [ <double 0.000000e+00, double -0.000000e+00, double -0.000000e+00, double -0.000000e+00>, %vector.ph ], [ [[RED_NEXT:%.*]], %vector.body ]
; CHECK-UNORDERED:    [[VECTOR_RECUR:%.*]] = phi <4 x double> [ <double poison, double poison, double poison, double 0.000000e+00>, %vector.ph ], [ [[FOR_NEXT:%.*]], %vector.body ]
; CHECK-UNORDERED:    [[FOR_NEXT]] = sitofp <4 x i32> %vec.ind to <4 x double>
; CHECK-UNORDERED:    [[TMP1:%.*]] = shufflevector <4 x double> [[VECTOR_RECUR]], <4 x double> [[FOR_NEXT]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-UNORDERED:    [[RED_NEXT]] = fadd <4 x double> [[TMP1]], [[RED]]
; CHECK-UNORDERED:  middle.block:
; CHECK-UNORDERED:    [[RDX:%.*]] = call double @llvm.vector.reduce.fadd.v4f64(double -0.000000e+00, <4 x double> [[RED_NEXT]])
; CHECK-UNORDERED:  scalar.ph:
; CHECK-UNORDERED:    [[BC_MERGE_RDX:%.*]] = phi double [ 0.000000e+00, %entry ], [ [[RDX]], %middle.block ]
;
; CHECK-NOT-VECTORIZED-LABEL: @reduction_increment_by_first_order_recurrence(
; CHECK-NOT-VECTORIZED-NOT: vector.body
;
entry:
  br label %loop

loop:
  %red = phi double [ 0.0, %entry ], [ %red.next, %loop ]
  %for = phi double [ 0.0, %entry ], [ %for.next, %loop ]
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %red.next = fadd double %for, %red
  %for.next = sitofp i32 %iv to double
  %iv.next = add nsw i32 %iv, 1
  %ec = icmp eq i32 %iv.next, 0
  br i1 %ec, label %exit, label %loop, !llvm.loop !13

exit:
  %res = phi double [ %red.next, %loop ]
  ret double %res
}

; We should not mark the fadd as an ordered reduction here as there are
; more than 2 uses of the instruction
define float @fadd_multiple_use(i64 %n) {
; CHECK-ORDERED-LABEL: @fadd_multiple_use
; CHECK-ORDERED-LABEL-NOT: vector.body

; CHECK-UNORDERED-LABEL: @fadd_multiple_use
; CHECK-UNORDERED-LABEL-NOT: vector.body

; CHECK-NOT-VECTORIZED-LABEL: @fadd_multiple_use
; CHECK-NOT-VECTORIZED-NOT: vector.body

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next2, %bb2 ]
  %red = phi float [ 0.0, %entry ], [ %fadd, %bb2 ]
  %phi1 = phi i64 [ 0, %entry ], [ %iv.next, %bb2 ]
  %fadd = fadd float %red, 1.000000e+00
  %iv.next = add nsw i64 %phi1, 1
  %cmp = icmp ult i64 %iv, %n
  br i1 %cmp, label %bb2, label %bb1

bb1:
  %phi2 = phi float [ %fadd, %for.body ]
  ret float %phi2

bb2:
  %iv.next2 = add nuw nsw i64 %iv, 1
  br i1 false, label %for.end, label %for.body

for.end:
  %phi3 = phi float [ %fadd, %bb2 ]
  ret float %phi3
}

; Test case where the loop has a call to the llvm.fmuladd intrinsic.
define float @fmuladd_strict(float* %a, float* %b, i64 %n) {
; CHECK-ORDERED-LABEL: @fmuladd_strict
; CHECK-ORDERED: vector.body:
; CHECK-ORDERED: [[VEC_PHI:%.*]] = phi float [ 0.000000e+00, %vector.ph ], [ [[RDX3:%.*]], %vector.body ]
; CHECK-ORDERED: [[WIDE_LOAD:%.*]] = load <8 x float>, <8 x float>*
; CHECK-ORDERED: [[WIDE_LOAD1:%.*]] = load <8 x float>, <8 x float>*
; CHECK-ORDERED: [[WIDE_LOAD2:%.*]] = load <8 x float>, <8 x float>*
; CHECK-ORDERED: [[WIDE_LOAD3:%.*]] = load <8 x float>, <8 x float>*
; CHECK-ORDERED: [[WIDE_LOAD4:%.*]] = load <8 x float>, <8 x float>*
; CHECK-ORDERED: [[WIDE_LOAD5:%.*]] = load <8 x float>, <8 x float>*
; CHECK-ORDERED: [[WIDE_LOAD6:%.*]] = load <8 x float>, <8 x float>*
; CHECK-ORDERED: [[WIDE_LOAD7:%.*]] = load <8 x float>, <8 x float>*
; CHECK-ORDERED: [[FMUL:%.*]] = fmul <8 x float> [[WIDE_LOAD]], [[WIDE_LOAD4]]
; CHECK-ORDERED: [[FMUL1:%.*]] = fmul <8 x float> [[WIDE_LOAD1]], [[WIDE_LOAD5]]
; CHECK-ORDERED: [[FMUL2:%.*]] = fmul <8 x float> [[WIDE_LOAD2]], [[WIDE_LOAD6]]
; CHECK-ORDERED: [[FMUL3:%.*]] = fmul <8 x float> [[WIDE_LOAD3]], [[WIDE_LOAD7]]
; CHECK-ORDERED: [[RDX:%.*]] = call float @llvm.vector.reduce.fadd.v8f32(float [[VEC_PHI]], <8 x float> [[FMUL]])
; CHECK-ORDERED: [[RDX1:%.*]] = call float @llvm.vector.reduce.fadd.v8f32(float [[RDX]], <8 x float> [[FMUL1]])
; CHECK-ORDERED: [[RDX2:%.*]] = call float @llvm.vector.reduce.fadd.v8f32(float [[RDX1]], <8 x float> [[FMUL2]])
; CHECK-ORDERED: [[RDX3]] = call float @llvm.vector.reduce.fadd.v8f32(float [[RDX2]], <8 x float> [[FMUL3]])
; CHECK-ORDERED: for.body:
; CHECK-ORDERED: [[SUM_07:%.*]] = phi float [ {{.*}}, %scalar.ph ], [ [[MULADD:%.*]], %for.body ]
; CHECK-ORDERED: [[LOAD:%.*]] = load float, float*
; CHECK-ORDERED: [[LOAD1:%.*]] = load float, float*
; CHECK-ORDERED: [[MULADD]] = tail call float @llvm.fmuladd.f32(float [[LOAD]], float [[LOAD1]], float [[SUM_07]])
; CHECK-ORDERED: for.end
; CHECK-ORDERED: [[RES:%.*]] = phi float [ [[MULADD]], %for.body ], [ [[RDX3]], %middle.block ]

; CHECK-UNORDERED-LABEL: @fmuladd_strict
; CHECK-UNORDERED: vector.body:
; CHECK-UNORDERED: [[VEC_PHI:%.*]] = phi <8 x float> [ <float 0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %vector.ph ], [ [[FMULADD:%.*]], %vector.body ]
; CHECK-UNORDERED: [[WIDE_LOAD:%.*]] = load <8 x float>, <8 x float>*
; CHECK-UNORDERED: [[WIDE_LOAD1:%.*]] = load <8 x float>, <8 x float>*
; CHECK-UNORDERED: [[WIDE_LOAD2:%.*]] = load <8 x float>, <8 x float>*
; CHECK-UNORDERED: [[WIDE_LOAD3:%.*]] = load <8 x float>, <8 x float>*
; CHECK-UNORDERED: [[WIDE_LOAD4:%.*]] = load <8 x float>, <8 x float>*
; CHECK-UNORDERED: [[FMULADD]] = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> [[WIDE_LOAD]], <8 x float> [[WIDE_LOAD4]], <8 x float> [[VEC_PHI]])
; CHECK-UNORDERED-NOT: llvm.vector.reduce.fadd
; CHECK-UNORDERED: middle.block:
; CHECK-UNORDERED: [[BIN_RDX1:%.*]] = fadd <8 x float>
; CHECK-UNORDERED: [[BIN_RDX2:%.*]] = fadd <8 x float>
; CHECK-UNORDERED: [[BIN_RDX3:%.*]] = fadd <8 x float>
; CHECK-UNORDERED: [[RDX:%.*]] = call float @llvm.vector.reduce.fadd.v8f32(float -0.000000e+00, <8 x float> [[BIN_RDX3]])
; CHECK-UNORDERED: for.body:
; CHECK-UNORDERED: [[SUM_07:%.*]] = phi float [ {{.*}}, %scalar.ph ], [  [[MULADD:%.*]], %for.body ]
; CHECK-UNORDERED: [[LOAD:%.*]] = load float, float*
; CHECK-UNORDERED: [[LOAD2:%.*]] = load float, float*
; CHECK-UNORDERED: [[MULADD]] = tail call float @llvm.fmuladd.f32(float [[LOAD]], float [[LOAD2]], float [[SUM_07]])
; CHECK-UNORDERED: for.end:
; CHECK-UNORDERED: [[RES:%.*]] = phi float [ [[MULADD]], %for.body ], [ [[RDX]], %middle.block ]
; CHECK-UNORDERED: ret float [[RES]]

; CHECK-NOT-VECTORIZED-LABEL: @fmuladd_strict
; CHECK-NOT-VECTORIZED-NOT: vector.body

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi float [ 0.000000e+00, %entry ], [ %muladd, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %a, i64 %iv
  %0 = load float, float* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, float* %b, i64 %iv
  %1 = load float, float* %arrayidx2, align 4
  %muladd = tail call float @llvm.fmuladd.f32(float %0, float %1, float %sum.07)
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !1

for.end:
  ret float %muladd
}

; Test reductions for a VF of 1 and a UF > 1 where the loop has a call to the llvm.fmuladd intrinsic.
define float @fmuladd_scalar_vf(float* %a, float* %b, i64 %n) {
; CHECK-ORDERED-LABEL: @fmuladd_scalar_vf
; CHECK-ORDERED: vector.body:
; CHECK-ORDERED: [[VEC_PHI:%.*]] = phi float [ 0.000000e+00, %vector.ph ], [ [[FADD3:%.*]], %vector.body ]
; CHECK-ORDERED: [[LOAD:%.*]] = load float, float*
; CHECK-ORDERED: [[LOAD1:%.*]] = load float, float*
; CHECK-ORDERED: [[LOAD2:%.*]] = load float, float*
; CHECK-ORDERED: [[LOAD3:%.*]] = load float, float*
; CHECK-ORDERED: [[LOAD4:%.*]] = load float, float*
; CHECK-ORDERED: [[LOAD5:%.*]] = load float, float*
; CHECK-ORDERED: [[LOAD6:%.*]] = load float, float*
; CHECK-ORDERED: [[LOAD7:%.*]] = load float, float*
; CHECK-ORDERED: [[FMUL:%.*]] = fmul float [[LOAD]], [[LOAD4]]
; CHECK-ORDERED: [[FMUL1:%.*]] = fmul float [[LOAD1]], [[LOAD5]]
; CHECK-ORDERED: [[FMUL2:%.*]] = fmul float [[LOAD2]], [[LOAD6]]
; CHECK-ORDERED: [[FMUL3:%.*]] = fmul float [[LOAD3]], [[LOAD7]]
; CHECK-ORDERED: [[FADD:%.*]] = fadd float [[VEC_PHI]], [[FMUL]]
; CHECK-ORDERED: [[FADD1:%.*]] = fadd float [[FADD]], [[FMUL1]]
; CHECK-ORDERED: [[FADD2:%.*]] = fadd float [[FADD1]], [[FMUL2]]
; CHECK-ORDERED: [[FADD3]] = fadd float [[FADD2]], [[FMUL3]]
; CHECK-ORDERED-NOT: llvm.vector.reduce.fadd
; CHECK-ORDERED: scalar.ph
; CHECK-ORDERED: [[MERGE_RDX:%.*]] = phi float [ 0.000000e+00, %entry ], [ [[FADD3]], %middle.block ]
; CHECK-ORDERED: for.body
; CHECK-ORDERED: [[SUM_07:%.*]] = phi float [ [[MERGE_RDX]], %scalar.ph ], [ [[MULADD:%.*]], %for.body ]
; CHECK-ORDERED: [[LOAD8:%.*]] = load float, float*
; CHECK-ORDERED: [[LOAD9:%.*]] = load float, float*
; CHECK-ORDERED: [[MULADD]] = tail call float @llvm.fmuladd.f32(float [[LOAD8]], float [[LOAD9]], float [[SUM_07]])
; CHECK-ORDERED: for.end
; CHECK-ORDERED: [[RES:%.*]] = phi float [ [[MULADD]], %for.body ], [ [[FADD3]], %middle.block ]
; CHECK-ORDERED: ret float [[RES]]

; CHECK-UNORDERED-LABEL: @fmuladd_scalar_vf
; CHECK-UNORDERED: vector.body:
; CHECK-UNORDERED: [[VEC_PHI:%.*]] = phi float [ 0.000000e+00, %vector.ph ], [ [[FMULADD:%.*]], %vector.body ]
; CHECK-UNORDERED: [[VEC_PHI1:%.*]] = phi float [ -0.000000e+00, %vector.ph ], [ [[FMULADD1:%.*]], %vector.body ]
; CHECK-UNORDERED: [[VEC_PHI2:%.*]] = phi float [ -0.000000e+00, %vector.ph ], [ [[FMULADD2:%.*]], %vector.body ]
; CHECK-UNORDERED: [[VEC_PHI3:%.*]] = phi float [ -0.000000e+00, %vector.ph ], [ [[FMULADD3:%.*]], %vector.body ]
; CHECK-UNORDERED: [[LOAD:%.*]] = load float, float*
; CHECK-UNORDERED: [[LOAD1:%.*]] = load float, float*
; CHECK-UNORDERED: [[LOAD2:%.*]] = load float, float*
; CHECK-UNORDERED: [[LOAD3:%.*]] = load float, float*
; CHECK-UNORDERED: [[LOAD4:%.*]] = load float, float*
; CHECK-UNORDERED: [[LOAD5:%.*]] = load float, float*
; CHECK-UNORDERED: [[LOAD6:%.*]] = load float, float*
; CHECK-UNORDERED: [[LOAD7:%.*]] = load float, float*
; CHECK-UNORDERED: [[FMULADD]] = tail call float @llvm.fmuladd.f32(float [[LOAD]], float [[LOAD4]], float [[VEC_PHI]])
; CHECK-UNORDERED: [[FMULADD1]] = tail call float @llvm.fmuladd.f32(float [[LOAD1]], float [[LOAD5]], float [[VEC_PHI1]])
; CHECK-UNORDERED: [[FMULADD2]] = tail call float @llvm.fmuladd.f32(float [[LOAD2]], float [[LOAD6]], float [[VEC_PHI2]])
; CHECK-UNORDERED: [[FMULADD3]] = tail call float @llvm.fmuladd.f32(float [[LOAD3]], float [[LOAD7]], float [[VEC_PHI3]])
; CHECK-UNORDERED-NOT: llvm.vector.reduce.fadd
; CHECK-UNORDERED: middle.block:
; CHECK-UNORDERED: [[BIN_RDX:%.*]] = fadd float [[FMULADD1]], [[FMULADD]]
; CHECK-UNORDERED: [[BIN_RDX1:%.*]] = fadd float [[FMULADD2]], [[BIN_RDX]]
; CHECK-UNORDERED: [[BIN_RDX2:%.*]] = fadd float [[FMULADD3]], [[BIN_RDX1]]
; CHECK-UNORDERED: scalar.ph:
; CHECK-UNORDERED: [[MERGE_RDX:%.*]] = phi float [ 0.000000e+00, %entry ], [ [[BIN_RDX2]], %middle.block ]
; CHECK-UNORDERED: for.body:
; CHECK-UNORDERED: [[SUM_07:%.*]] = phi float [ [[MERGE_RDX]], %scalar.ph ], [ [[MULADD:%.*]], %for.body ]
; CHECK-UNORDERED: [[LOAD8:%.*]] = load float, float*
; CHECK-UNORDERED: [[LOAD9:%.*]] = load float, float*
; CHECK-UNORDERED: [[MULADD]] = tail call float @llvm.fmuladd.f32(float [[LOAD8]], float [[LOAD9]], float [[SUM_07]])
; CHECK-UNORDERED: for.end:
; CHECK-UNORDERED: [[RES:%.*]] = phi float [ [[MULADD]], %for.body ], [ [[BIN_RDX2]], %middle.block ]
; CHECK-UNORDERED: ret float [[RES]]

; CHECK-NOT-VECTORIZED-LABEL: @fmuladd_scalar_vf
; CHECK-NOT-VECTORIZED-NOT: vector.body

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi float [ 0.000000e+00, %entry ], [ %muladd, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %a, i64 %iv
  %0 = load float, float* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, float* %b, i64 %iv
  %1 = load float, float* %arrayidx2, align 4
  %muladd = tail call float @llvm.fmuladd.f32(float %0, float %1, float %sum.07)
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !4

for.end:
  ret float %muladd
}

; Test case where the reduction phi is one of the mul operands of the fmuladd.
define float @fmuladd_phi_is_mul_operand(float* %a, float* %b, i64 %n) {
; CHECK-ORDERED-LABEL: @fmuladd_phi_is_mul_operand
; CHECK-ORDERED-NOT: vector.body

; CHECK-UNORDERED-LABEL: @fmuladd_phi_is_mul_operand
; CHECK-UNORDERED-NOT: vector.body

; CHECK-NOT-VECTORIZED-LABEL: @fmuladd_phi_is_mul_operand
; CHECK-NOT-VECTORIZED-NOT: vector.body

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi float [ 0.000000e+00, %entry ], [ %muladd, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %a, i64 %iv
  %0 = load float, float* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, float* %b, i64 %iv
  %1 = load float, float* %arrayidx2, align 4
  %muladd = tail call float @llvm.fmuladd.f32(float %sum.07, float %0, float %1)
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !1

for.end:
  ret float %muladd
}

; Test case where the reduction phi is two operands of the fmuladd.
define float @fmuladd_phi_is_two_operands(float* %a, i64 %n) {
; CHECK-ORDERED-LABEL: @fmuladd_phi_is_two_operands
; CHECK-ORDERED-NOT: vector.body

; CHECK-UNORDERED-LABEL: @fmuladd_phi_is_two_operands
; CHECK-UNORDERED-NOT: vector.body

; CHECK-NOT-VECTORIZED-LABEL: @fmuladd_phi_is_two_operands
; CHECK-NOT-VECTORIZED-NOT: vector.body

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi float [ 0.000000e+00, %entry ], [ %muladd, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %a, i64 %iv
  %0 = load float, float* %arrayidx, align 4
  %muladd = tail call float @llvm.fmuladd.f32(float %sum.07, float %0, float %sum.07)
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !1

for.end:
  ret float %muladd
}

; Test case with multiple calls to llvm.fmuladd, which is not safe to reorder
; so is only vectorized in the unordered (fast) case.
define float @fmuladd_multiple(float* %a, float* %b, i64 %n) {
; CHECK-ORDERED-LABEL: @fmuladd_multiple
; CHECK-ORDERED-NOT: vector.body:

; CHECK-UNORDERED-LABEL: @fmuladd_multiple
; CHECK-UNORDERED: vector.body:
; CHECK-UNORDERED: [[VEC_PHI:%.*]] = phi <8 x float> [ <float 0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %vector.ph ], [ [[FMULADD2:%.*]], %vector.body ]
; CHECK-UNORDERED: [[WIDE_LOAD:%.*]] = load <8 x float>, <8 x float>*
; CHECK-UNORDERED: [[WIDE_LOAD1:%.*]] = load <8 x float>, <8 x float>*
; CHECK-UNORDERED: [[WIDE_LOAD2:%.*]] = load <8 x float>, <8 x float>*
; CHECK-UNORDERED: [[WIDE_LOAD3:%.*]] = load <8 x float>, <8 x float>*
; CHECK-UNORDERED: [[WIDE_LOAD4:%.*]] = load <8 x float>, <8 x float>*
; CHECK-UNORDERED: [[FMULADD:%.*]] = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> [[WIDE_LOAD]], <8 x float> [[WIDE_LOAD4]], <8 x float> [[VEC_PHI]])
; CHECK-UNORDERED: [[FMULADD2]] = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> [[WIDE_LOAD]], <8 x float> [[WIDE_LOAD4]], <8 x float> [[FMULADD]])
; CHECK-UNORDERED-NOT: llvm.vector.reduce.fadd
; CHECK-UNORDERED: middle.block:
; CHECK-UNORDERED: [[BIN_RDX1:%.*]] = fadd <8 x float>
; CHECK-UNORDERED: [[BIN_RDX2:%.*]] = fadd <8 x float>
; CHECK-UNORDERED: [[BIN_RDX3:%.*]] = fadd <8 x float>
; CHECK-UNORDERED: [[RDX:%.*]] = call float @llvm.vector.reduce.fadd.v8f32(float -0.000000e+00, <8 x float> [[BIN_RDX3]])
; CHECK-UNORDERED: for.body:
; CHECK-UNORDERED: [[SUM_07:%.*]] = phi float [ {{.*}}, %scalar.ph ], [ [[MULADD2:%.*]], %for.body ]
; CHECK-UNORDERED: [[LOAD:%.*]] = load float, float*
; CHECK-UNORDERED: [[LOAD2:%.*]] = load float, float*
; CHECK-UNORDERED: [[MULADD:%.*]] = tail call float @llvm.fmuladd.f32(float [[LOAD]], float [[LOAD2]], float [[SUM_07]])
; CHECK-UNORDERED: [[MULADD2]] = tail call float @llvm.fmuladd.f32(float [[LOAD]], float [[LOAD2]], float [[MULADD]])
; CHECK-UNORDERED: for.end:
; CHECK-UNORDERED: [[RES:%.*]] = phi float [ [[MULADD2]], %for.body ], [ [[RDX]], %middle.block ]
; CHECK-UNORDERED: ret float [[RES]]

; CHECK-NOT-VECTORIZED-LABEL: @fmuladd_multiple
; CHECK-NOT-VECTORIZED-NOT: vector.body:

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi float [ 0.000000e+00, %entry ], [ %muladd2, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %a, i64 %iv
  %0 = load float, float* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, float* %b, i64 %iv
  %1 = load float, float* %arrayidx2, align 4
  %muladd = tail call float @llvm.fmuladd.f32(float %0, float %1, float %sum.07)
  %muladd2 = tail call float @llvm.fmuladd.f32(float %0, float %1, float %muladd)
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !1

for.end:
  ret float %muladd2
}

; Same as above but the first fmuladd is one of the mul operands of the second fmuladd.
define float @multiple_fmuladds_mul_operand(float* %a, float* %b, i64 %n) {
; CHECK-ORDERED-LABEL: @multiple_fmuladds_mul_operand
; CHECK-ORDERED-NOT: vector.body

; CHECK-UNORDERED-LABEL: @multiple_fmuladds_mul_operand
; CHECK-UNORDERED-NOT: vector.body

; CHECK-NOT-VECTORIZED-LABEL: @multiple_fmuladds_mul_operand
; CHECK-NOT-VECTORIZED-NOT: vector.body

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi float [ 0.000000e+00, %entry ], [ %muladd2, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %a, i64 %iv
  %0 = load float, float* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, float* %b, i64 %iv
  %1 = load float, float* %arrayidx2, align 4
  %muladd = tail call float @llvm.fmuladd.f32(float %0, float %1, float %sum.07)
  %muladd2 = tail call float @llvm.fmuladd.f32(float %0, float %muladd, float %1)
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !1

for.end:
  ret float %muladd2
}

; Same as above but the first fmuladd is two of the operands of the second fmuladd.
define float @multiple_fmuladds_two_operands(float* %a, float* %b, i64 %n) {
; CHECK-ORDERED-LABEL: @multiple_fmuladds_two_operands
; CHECK-ORDERED-NOT: vector.body

; CHECK-UNORDERED-LABEL: @multiple_fmuladds_two_operands
; CHECK-UNORDERED-NOT: vector.body

; CHECK-NOT-VECTORIZED-LABEL: @multiple_fmuladds_two_operands
; CHECK-NOT-VECTORIZED-NOT: vector.body

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi float [ 0.000000e+00, %entry ], [ %muladd2, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %a, i64 %iv
  %0 = load float, float* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, float* %b, i64 %iv
  %1 = load float, float* %arrayidx2, align 4
  %muladd = tail call float @llvm.fmuladd.f32(float %0, float %1, float %sum.07)
  %muladd2 = tail call float @llvm.fmuladd.f32(float %0, float %muladd, float %muladd)
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !1

for.end:
  ret float %muladd2
}

declare float @llvm.fmuladd.f32(float, float, float)

; Test case with invariant store where fadd is strict.
define void @reduction_store_to_invariant_address(float* %dst, float* readonly %src) {
; CHECK-ORDERED-LABEL: @reduction_store_to_invariant_address(
; CHECK-ORDERED: entry
; CHECK-ORDERED: %[[DEST_PTR:.*]] = getelementptr inbounds float, float* %dst, i64 42
; CHECK-ORDERED: vector.body
; CHECK-ORDERED: %[[VEC_PHI:.*]] = phi float [ 0.000000e+00, %vector.ph ], [ %[[RDX:.*]], %vector.body ]
; CHECK-ORDERED: %[[LOAD_VEC:.*]] = load <8 x float>, <8 x float>*
; CHECK-ORDERED: %[[RDX:.*]] = call float @llvm.vector.reduce.fadd.v8f32(float %[[VEC_PHI]], <8 x float> %[[LOAD_VEC]])
; CHECK-ORDERED: middle.block
; CHECK-ORDERED: store float %[[RDX]], float* %[[DEST_PTR]]
; CHECK-ORDERED: for.body
; CHECK-ORDERED: %[[LOAD:.*]] = load float, float*
; CHECK-ORDERED: %[[FADD:.*]] = fadd float %{{.*}}, %[[LOAD]]
; CHECK-ORDERED: store float %[[FADD]], float* %[[DEST_PTR]]

; CHECK-UNORDERED-LABEL: @reduction_store_to_invariant_address(
; CHECK-UNORDERED: entry
; CHECK-UNORDERED: %[[DEST_PTR:.*]] = getelementptr inbounds float, float* %dst, i64 42
; CHECK-UNORDERED: vector.body
; CHECK-UNORDERED: %[[VEC_PHI:.*]] = phi <8 x float> [ <float 0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %vector.ph ], [ %[[FADD_VEC:.*]], %vector.body ]
; CHECK-UNORDERED: %[[LOAD_VEC:.*]] = load <8 x float>, <8 x float>*
; CHECK-UNORDERED: %[[FADD_VEC]] = fadd <8 x float> %[[VEC_PHI]], %[[LOAD_VEC]]
; CHECK-UNORDERED-NOT: call float @llvm.vector.reduce.fadd
; CHECK-UNORDERED: middle.block
; CHECK-UNORDERED: %[[RDX:.*]] = call float @llvm.vector.reduce.fadd.v8f32(float -0.000000e+00, <8 x float> %[[FADD_VEC]])
; CHECK-UNORDERED: store float %[[RDX]], float* %[[DEST_PTR]]
; CHECK-UNORDERED: for.body
; CHECK-UNORDERED: %[[LOAD:.*]] = load float, float*
; CHECK-UNORDERED: %[[FADD:.*]] = fadd float {{.*}}, %[[LOAD]]
; CHECK-UNORDERED: store float %[[FADD]], float* %[[DEST_PTR]]

; CHECK-NOT-VECTORIZED-LABEL: @reduction_store_to_invariant_address(
; CHECK-NOT-VECTORIZED-NOT: vector.body

entry:
  %arrayidx = getelementptr inbounds float, float* %dst, i64 42
  store float 0.000000e+00, float* %arrayidx, align 4
  br label %for.body

for.body:
  %0 = phi float [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx1 = getelementptr inbounds float, float* %src, i64 %indvars.iv
  %1 = load float, float* %arrayidx1, align 4
  %add = fadd float %0, %1
  store float %add, float* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !llvm.loop !0

for.cond.cleanup:
  ret void
}

!0 = distinct !{!0, !5, !9, !11}
!1 = distinct !{!1, !5, !10, !11}
!2 = distinct !{!2, !6, !9, !11}
!3 = distinct !{!3, !7, !9, !11, !12}
!4 = distinct !{!4, !8, !10, !11}
!5 = !{!"llvm.loop.vectorize.width", i32 8}
!6 = !{!"llvm.loop.vectorize.width", i32 4}
!7 = !{!"llvm.loop.vectorize.width", i32 2}
!8 = !{!"llvm.loop.vectorize.width", i32 1}
!9 = !{!"llvm.loop.interleave.count", i32 1}
!10 = !{!"llvm.loop.interleave.count", i32 4}
!11 = !{!"llvm.loop.vectorize.enable", i1 true}
!12 = !{!"llvm.loop.vectorize.predicate.enable", i1 true}
!13 = distinct !{!13, !6, !9, !11}
