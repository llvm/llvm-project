; RUN: opt < %s -loop-vectorize -mtriple aarch64-unknown-linux-gnu -mattr=+sve -prefer-predicate-over-epilogue=scalar-epilogue \
; RUN:   -force-ordered-reductions=false -hints-allow-reordering=false -S | FileCheck %s --check-prefix=CHECK-NOT-VECTORIZED
; RUN: opt < %s -loop-vectorize -mtriple aarch64-unknown-linux-gnu -mattr=+sve -prefer-predicate-over-epilogue=scalar-epilogue \
; RUN:   -force-ordered-reductions=false -hints-allow-reordering=true  -S | FileCheck %s --check-prefix=CHECK-UNORDERED
; RUN: opt < %s -loop-vectorize -mtriple aarch64-unknown-linux-gnu -mattr=+sve -prefer-predicate-over-epilogue=scalar-epilogue \
; RUN:   -force-ordered-reductions=true  -hints-allow-reordering=false -S | FileCheck %s --check-prefix=CHECK-ORDERED
; RUN: opt < %s -loop-vectorize -mtriple aarch64-unknown-linux-gnu -mattr=+sve -prefer-predicate-over-epilogue=scalar-epilogue \
; RUN:   -force-ordered-reductions=true  -hints-allow-reordering=true  -S | FileCheck %s --check-prefix=CHECK-UNORDERED
; RUN: opt < %s -loop-vectorize -mtriple aarch64-unknown-linux-gnu -mattr=+sve -prefer-predicate-over-epilogue=scalar-epilogue \
; RUN:   -hints-allow-reordering=false -S | FileCheck %s --check-prefix=CHECK-ORDERED
; RUN: opt < %s -loop-vectorize -mtriple aarch64-unknown-linux-gnu -mattr=+sve -prefer-predicate-over-epilogue=predicate-else-scalar-epilogue \
; RUN:   -hints-allow-reordering=false -S | FileCheck %s --check-prefix=CHECK-ORDERED-TF

define float @fadd_strict(float* noalias nocapture readonly %a, i64 %n) #0 {
; CHECK-ORDERED-LABEL: @fadd_strict
; CHECK-ORDERED: vector.body:
; CHECK-ORDERED: %[[VEC_PHI:.*]] = phi float [ 0.000000e+00, %vector.ph ], [ %[[RDX:.*]], %vector.body ]
; CHECK-ORDERED: %[[LOAD:.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-ORDERED: %[[RDX]] = call float @llvm.vector.reduce.fadd.nxv8f32(float %[[VEC_PHI]], <vscale x 8 x float> %[[LOAD]])
; CHECK-ORDERED: for.end
; CHECK-ORDERED: %[[PHI:.*]] = phi float [ %[[SCALAR:.*]], %for.body ], [ %[[RDX]], %middle.block ]
; CHECK-ORDERED: ret float %[[PHI]]

; CHECK-ORDERED-TF-LABEL: @fadd_strict
; CHECK-ORDERED-TF: vector.body:
; CHECK-ORDERED-TF: %[[ACTIVE_LANE_MASK:.*]] = phi <vscale x 8 x i1>
; CHECK-ORDERED-TF: %[[VEC_PHI:.*]] = phi float [ 0.000000e+00, %vector.ph ], [ %[[RDX:.*]], %vector.body ]
; CHECK-ORDERED-TF: %[[LOAD:.*]] = call <vscale x 8 x float> @llvm.masked.load.nxv8f32.p0nxv8f32(<vscale x 8 x float>*
; CHECK-ORDERED-TF: %[[SEL:.*]] = select <vscale x 8 x i1> %[[ACTIVE_LANE_MASK]], <vscale x 8 x float> %[[LOAD]], <vscale x 8 x float> shufflevector (<vscale x 8 x float> insertelement (<vscale x 8 x float> poison, float -0.000000e+00, i32 0), <vscale x 8 x float> poison, <vscale x 8 x i32> zeroinitializer)
; CHECK-ORDERED-TF: %[[RDX]] = call float @llvm.vector.reduce.fadd.nxv8f32(float %[[VEC_PHI]], <vscale x 8 x float> %[[SEL]])
; CHECK-ORDERED-TF: for.end
; CHECK-ORDERED-TF: %[[PHI:.*]] = phi float [ %[[SCALAR:.*]], %for.body ], [ %[[RDX]], %middle.block ]
; CHECK-ORDERED-TF: ret float %[[PHI]]

; CHECK-UNORDERED-LABEL: @fadd_strict
; CHECK-UNORDERED: vector.body
; CHECK-UNORDERED: %[[VEC_PHI:.*]] = phi <vscale x 8 x float> [ insertelement (<vscale x 8 x float> shufflevector (<vscale x 8 x float> insertelement (<vscale x 8 x float> poison, float -0.000000e+00, i32 0), <vscale x 8 x float> poison, <vscale x 8 x i32> zeroinitializer), float 0.000000e+00, i32 0), %vector.ph ], [ %[[FADD_VEC:.*]], %vector.body ]
; CHECK-UNORDERED: %[[LOAD_VEC:.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-UNORDERED: %[[FADD_VEC]] = fadd <vscale x 8 x float> %[[LOAD_VEC]], %[[VEC_PHI]]
; CHECK-UNORDERED-NOT: call float @llvm.vector.reduce.fadd
; CHECK-UNORDERED: middle.block
; CHECK-UNORDERED: %[[RDX:.*]] = call float @llvm.vector.reduce.fadd.nxv8f32(float -0.000000e+00, <vscale x 8 x float> %[[FADD_VEC]])
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

define float @fadd_strict_unroll(float* noalias nocapture readonly %a, i64 %n) #0 {
; CHECK-ORDERED-LABEL: @fadd_strict_unroll
; CHECK-ORDERED: vector.body:
; CHECK-ORDERED: %[[VEC_PHI1:.*]] = phi float [ 0.000000e+00, %vector.ph ], [ %[[RDX4:.*]], %vector.body ]
; CHECK-ORDERED-NOT: phi float [ 0.000000e+00, %vector.ph ], [ %[[RDX4]], %vector.body ]
; CHECK-ORDERED: %[[LOAD1:.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-ORDERED: %[[LOAD2:.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-ORDERED: %[[LOAD3:.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-ORDERED: %[[LOAD4:.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-ORDERED: %[[RDX1:.*]] = call float @llvm.vector.reduce.fadd.nxv8f32(float %[[VEC_PHI1]], <vscale x 8 x float> %[[LOAD1]])
; CHECK-ORDERED: %[[RDX2:.*]] = call float @llvm.vector.reduce.fadd.nxv8f32(float %[[RDX1]], <vscale x 8 x float> %[[LOAD2]])
; CHECK-ORDERED: %[[RDX3:.*]] = call float @llvm.vector.reduce.fadd.nxv8f32(float %[[RDX2]], <vscale x 8 x float> %[[LOAD3]])
; CHECK-ORDERED: %[[RDX4]] = call float @llvm.vector.reduce.fadd.nxv8f32(float %[[RDX3]], <vscale x 8 x float> %[[LOAD4]])
; CHECK-ORDERED: for.end
; CHECK-ORDERED: %[[PHI:.*]] = phi float [ %[[SCALAR:.*]], %for.body ], [ %[[RDX4]], %middle.block ]
; CHECK-ORDERED: ret float %[[PHI]]

; CHECK-ORDERED-TF-LABEL: @fadd_strict_unroll
; CHECK-ORDERED-TF: vector.body:
; CHECK-ORDERED-TF: %[[ACTIVE_LANE_MASK1:.*]] = phi <vscale x 8 x i1>
; CHECK-ORDERED-TF: %[[ACTIVE_LANE_MASK2:.*]] = phi <vscale x 8 x i1>
; CHECK-ORDERED-TF: %[[ACTIVE_LANE_MASK3:.*]] = phi <vscale x 8 x i1>
; CHECK-ORDERED-TF: %[[ACTIVE_LANE_MASK4:.*]] = phi <vscale x 8 x i1>
; CHECK-ORDERED-TF: %[[VEC_PHI1:.*]] = phi float [ 0.000000e+00, %vector.ph ], [ %[[RDX4:.*]], %vector.body ]
; CHECK-ORDERED-TF-NOT: phi float [ 0.000000e+00, %vector.ph ], [ %[[RDX4]], %vector.body ]
; CHECK-ORDERED-TF: %[[LOAD1:.*]] = call <vscale x 8 x float> @llvm.masked.load.nxv8f32.p0nxv8f32(<vscale x 8 x float>*
; CHECK-ORDERED-TF: %[[LOAD2:.*]] = call <vscale x 8 x float> @llvm.masked.load.nxv8f32.p0nxv8f32(<vscale x 8 x float>*
; CHECK-ORDERED-TF: %[[LOAD3:.*]] = call <vscale x 8 x float> @llvm.masked.load.nxv8f32.p0nxv8f32(<vscale x 8 x float>*
; CHECK-ORDERED-TF: %[[LOAD4:.*]] = call <vscale x 8 x float> @llvm.masked.load.nxv8f32.p0nxv8f32(<vscale x 8 x float>*
; CHECK-ORDERED-TF: %[[SEL1:.*]] = select <vscale x 8 x i1> %[[ACTIVE_LANE_MASK1]], <vscale x 8 x float> %[[LOAD1]], <vscale x 8 x float> shufflevector (<vscale x 8 x float> insertelement (<vscale x 8 x float> poison, float -0.000000e+00, i32 0), <vscale x 8 x float> poison, <vscale x 8 x i32> zeroinitializer)
; CHECK-ORDERED-TF: %[[RDX1:.*]] = call float @llvm.vector.reduce.fadd.nxv8f32(float %[[VEC_PHI1]], <vscale x 8 x float> %[[SEL1]])
; CHECK-ORDERED-TF: %[[SEL2:.*]] = select <vscale x 8 x i1> %[[ACTIVE_LANE_MASK2]], <vscale x 8 x float> %[[LOAD2]], <vscale x 8 x float> shufflevector (<vscale x 8 x float> insertelement (<vscale x 8 x float> poison, float -0.000000e+00, i32 0), <vscale x 8 x float> poison, <vscale x 8 x i32> zeroinitializer)
; CHECK-ORDERED-TF: %[[RDX2:.*]] = call float @llvm.vector.reduce.fadd.nxv8f32(float %[[RDX1]], <vscale x 8 x float> %[[SEL2]])
; CHECK-ORDERED-TF: %[[SEL3:.*]] = select <vscale x 8 x i1> %[[ACTIVE_LANE_MASK3]], <vscale x 8 x float> %[[LOAD3]], <vscale x 8 x float> shufflevector (<vscale x 8 x float> insertelement (<vscale x 8 x float> poison, float -0.000000e+00, i32 0), <vscale x 8 x float> poison, <vscale x 8 x i32> zeroinitializer)
; CHECK-ORDERED-TF: %[[RDX3:.*]] = call float @llvm.vector.reduce.fadd.nxv8f32(float %[[RDX2]], <vscale x 8 x float> %[[SEL3]])
; CHECK-ORDERED-TF: %[[SEL4:.*]] = select <vscale x 8 x i1> %[[ACTIVE_LANE_MASK4]], <vscale x 8 x float> %[[LOAD4]], <vscale x 8 x float> shufflevector (<vscale x 8 x float> insertelement (<vscale x 8 x float> poison, float -0.000000e+00, i32 0), <vscale x 8 x float> poison, <vscale x 8 x i32> zeroinitializer)
; CHECK-ORDERED-TF: %[[RDX4]] = call float @llvm.vector.reduce.fadd.nxv8f32(float %[[RDX3]], <vscale x 8 x float> %[[SEL4]])
; CHECK-ORDERED-TF: for.end
; CHECK-ORDERED-TF: %[[PHI:.*]] = phi float [ %[[SCALAR:.*]], %for.body ], [ %[[RDX4]], %middle.block ]
; CHECK-ORDERED-TF: ret float %[[PHI]]

; CHECK-UNORDERED-LABEL: @fadd_strict_unroll
; CHECK-UNORDERED: vector.body
; CHECK-UNORDERED: %[[VEC_PHI1:.*]] = phi <vscale x 8 x float> [ insertelement (<vscale x 8 x float> shufflevector (<vscale x 8 x float> insertelement (<vscale x 8 x float> poison, float -0.000000e+00, i32 0), <vscale x 8 x float> poison, <vscale x 8 x i32> zeroinitializer), float 0.000000e+00, i32 0), %vector.ph ], [ %[[VEC_FADD1:.*]], %vector.body ]
; CHECK-UNORDERED: %[[VEC_PHI2:.*]] = phi <vscale x 8 x float> [ shufflevector (<vscale x 8 x float> insertelement (<vscale x 8 x float> poison, float -0.000000e+00, i32 0), <vscale x 8 x float> poison, <vscale x 8 x i32> zeroinitializer), %vector.ph ], [ %[[VEC_FADD2:.*]], %vector.body ]
; CHECK-UNORDERED: %[[VEC_PHI3:.*]] = phi <vscale x 8 x float> [ shufflevector (<vscale x 8 x float> insertelement (<vscale x 8 x float> poison, float -0.000000e+00, i32 0), <vscale x 8 x float> poison, <vscale x 8 x i32> zeroinitializer), %vector.ph ], [ %[[VEC_FADD3:.*]], %vector.body ]
; CHECK-UNORDERED: %[[VEC_PHI4:.*]] = phi <vscale x 8 x float> [ shufflevector (<vscale x 8 x float> insertelement (<vscale x 8 x float> poison, float -0.000000e+00, i32 0), <vscale x 8 x float> poison, <vscale x 8 x i32> zeroinitializer), %vector.ph ], [ %[[VEC_FADD4:.*]], %vector.body ]
; CHECK-UNORDERED: %[[VEC_LOAD1:.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-UNORDERED: %[[VEC_LOAD2:.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-UNORDERED: %[[VEC_LOAD3:.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-UNORDERED: %[[VEC_LOAD4:.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-UNORDERED: %[[VEC_FADD1]] = fadd <vscale x 8 x float> %[[VEC_LOAD1]], %[[VEC_PHI1]]
; CHECK-UNORDERED: %[[VEC_FADD2]] = fadd <vscale x 8 x float> %[[VEC_LOAD2]], %[[VEC_PHI2]]
; CHECK-UNORDERED: %[[VEC_FADD3]] = fadd <vscale x 8 x float> %[[VEC_LOAD3]], %[[VEC_PHI3]]
; CHECK-UNORDERED: %[[VEC_FADD4]] = fadd <vscale x 8 x float> %[[VEC_LOAD4]], %[[VEC_PHI4]]
; CHECK-UNORDERED-NOT: call float @llvm.vector.reduce.fadd
; CHECK-UNORDERED: middle.block
; CHECK-UNORDERED: %[[BIN_RDX1:.*]] = fadd <vscale x 8 x float> %[[VEC_FADD2]], %[[VEC_FADD1]]
; CHECK-UNORDERED: %[[BIN_RDX2:.*]] = fadd <vscale x 8 x float> %[[VEC_FADD3]], %[[BIN_RDX1]]
; CHECK-UNORDERED: %[[BIN_RDX3:.*]] = fadd <vscale x 8 x float> %[[VEC_FADD4]], %[[BIN_RDX2]]
; CHECK-UNORDERED: %[[RDX:.*]] = call float @llvm.vector.reduce.fadd.nxv8f32(float -0.000000e+00, <vscale x 8 x float> %[[BIN_RDX3]])
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

define void @fadd_strict_interleave(float* noalias nocapture readonly %a, float* noalias nocapture readonly %b, i64 %n) #0 {
; CHECK-ORDERED-LABEL: @fadd_strict_interleave
; CHECK-ORDERED: entry
; CHECK-ORDERED: %[[ARRAYIDX:.*]] = getelementptr inbounds float, float* %a, i64 1
; CHECK-ORDERED: %[[LOAD1:.*]] = load float, float* %a
; CHECK-ORDERED: %[[LOAD2:.*]] = load float, float* %[[ARRAYIDX]]
; CHECK-ORDERED: vector.ph
; CHECK-ORDERED: %[[STEPVEC1:.*]] = call <vscale x 4 x i64> @llvm.experimental.stepvector.nxv4i64()
; CHECK-ORDERED: %[[STEPVEC_ADD1:.*]] = add <vscale x 4 x i64> %[[STEPVEC1]], zeroinitializer
; CHECK-ORDERED: %[[STEPVEC_MUL:.*]] = mul <vscale x 4 x i64> %[[STEPVEC_ADD1]], shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> poison, i64 2, i32 0), <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-ORDERED: %[[INDUCTION:.*]] = add <vscale x 4 x i64> zeroinitializer, %[[STEPVEC_MUL]]
; CHECK-ORDERED: vector.body
; CHECK-ORDERED: %[[VEC_PHI2:.*]] = phi float [ %[[LOAD2]], %vector.ph ], [ %[[RDX2:.*]], %vector.body ]
; CHECK-ORDERED: %[[VEC_PHI1:.*]] = phi float [ %[[LOAD1]], %vector.ph ], [ %[[RDX1:.*]], %vector.body ]
; CHECK-ORDERED: %[[VEC_IND:.*]] = phi <vscale x 4 x i64> [ %[[INDUCTION]], %vector.ph ], [ {{.*}}, %vector.body ]
; CHECK-ORDERED: %[[GEP1:.*]] = getelementptr inbounds float, float* %b, <vscale x 4 x i64> %[[VEC_IND]]
; CHECK-ORDERED: %[[MGATHER1:.*]] = call <vscale x 4 x float> @llvm.masked.gather.nxv4f32.nxv4p0f32(<vscale x 4 x float*> %[[GEP1]], i32 4, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i32 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> poison)
; CHECK-ORDERED: %[[OR:.*]] = or <vscale x 4 x i64> %[[VEC_IND]], shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> poison, i64 1, i32 0), <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-ORDERED: %[[GEP2:.*]] = getelementptr inbounds float, float* %b, <vscale x 4 x i64> %[[OR]]
; CHECK-ORDERED: %[[MGATHER2:.*]] = call <vscale x 4 x float> @llvm.masked.gather.nxv4f32.nxv4p0f32(<vscale x 4 x float*> %[[GEP2]], i32 4, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i32 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> poison)
; CHECK-ORDERED: %[[RDX2]] = call float @llvm.vector.reduce.fadd.nxv4f32(float %[[VEC_PHI2]], <vscale x 4 x float> %[[MGATHER2]])
; CHECK-ORDERED: %[[RDX1]] = call float @llvm.vector.reduce.fadd.nxv4f32(float %[[VEC_PHI1]], <vscale x 4 x float> %[[MGATHER1]])

; CHECK-ORDERED-TF-LABEL: @fadd_strict_interleave
; CHECK-ORDERED-TF: entry
; CHECK-ORDERED-TF: %[[ARRAYIDX:.*]] = getelementptr inbounds float, float* %a, i64 1
; CHECK-ORDERED-TF: %[[LOAD1:.*]] = load float, float* %a
; CHECK-ORDERED-TF: %[[LOAD2:.*]] = load float, float* %[[ARRAYIDX]]
; CHECK-ORDERED-TF: vector.ph
; CHECK-ORDERED-TF: %[[STEPVEC1:.*]] = call <vscale x 4 x i64> @llvm.experimental.stepvector.nxv4i64()
; CHECK-ORDERED-TF: %[[STEPVEC_ADD1:.*]] = add <vscale x 4 x i64> %[[STEPVEC1]], zeroinitializer
; CHECK-ORDERED-TF: %[[STEPVEC_MUL:.*]] = mul <vscale x 4 x i64> %[[STEPVEC_ADD1]], shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> poison, i64 2, i32 0), <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-ORDERED-TF: %[[INDUCTION:.*]] = add <vscale x 4 x i64> zeroinitializer, %[[STEPVEC_MUL]]
; CHECK-ORDERED-TF: vector.body
; CHECK-ORDERED-TF: %[[ACTIVE_LANE_MASK:.*]] = phi <vscale x 4 x i1>
; CHECK-ORDERED-TF: %[[VEC_PHI2:.*]] = phi float [ %[[LOAD2]], %vector.ph ], [ %[[RDX2:.*]], %vector.body ]
; CHECK-ORDERED-TF: %[[VEC_PHI1:.*]] = phi float [ %[[LOAD1]], %vector.ph ], [ %[[RDX1:.*]], %vector.body ]
; CHECK-ORDERED-TF: %[[VEC_IND:.*]] = phi <vscale x 4 x i64> [ %[[INDUCTION]], %vector.ph ], [ {{.*}}, %vector.body ]
; CHECK-ORDERED-TF: %[[GEP1:.*]] = getelementptr inbounds float, float* %b, <vscale x 4 x i64> %[[VEC_IND]]
; CHECK-ORDERED-TF: %[[MGATHER1:.*]] = call <vscale x 4 x float> @llvm.masked.gather.nxv4f32.nxv4p0f32(<vscale x 4 x float*> %[[GEP1]], i32 4, <vscale x 4 x i1> %[[ACTIVE_LANE_MASK]], <vscale x 4 x float> poison)
; CHECK-ORDERED-TF: %[[OR:.*]] = or <vscale x 4 x i64> %[[VEC_IND]], shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> poison, i64 1, i32 0), <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-ORDERED-TF: %[[GEP2:.*]] = getelementptr inbounds float, float* %b, <vscale x 4 x i64> %[[OR]]
; CHECK-ORDERED-TF: %[[MGATHER2:.*]] = call <vscale x 4 x float> @llvm.masked.gather.nxv4f32.nxv4p0f32(<vscale x 4 x float*> %[[GEP2]], i32 4, <vscale x 4 x i1> %[[ACTIVE_LANE_MASK]], <vscale x 4 x float> poison)
; CHECK-ORDERED-TF: %[[SEL2:.*]] = select <vscale x 4 x i1> %[[ACTIVE_LANE_MASK]], <vscale x 4 x float> %[[MGATHER2]], <vscale x 4 x float> shufflevector (<vscale x 4 x float> insertelement (<vscale x 4 x float> poison, float -0.000000e+00, i32 0), <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-ORDERED-TF: %[[RDX2]] = call float @llvm.vector.reduce.fadd.nxv4f32(float %[[VEC_PHI2]], <vscale x 4 x float> %[[SEL2]])
; CHECK-ORDERED-TF: %[[SEL1:.*]] = select <vscale x 4 x i1> %[[ACTIVE_LANE_MASK]], <vscale x 4 x float> %[[MGATHER1]], <vscale x 4 x float> shufflevector (<vscale x 4 x float> insertelement (<vscale x 4 x float> poison, float -0.000000e+00, i32 0), <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-ORDERED-TF: %[[RDX1]] = call float @llvm.vector.reduce.fadd.nxv4f32(float %[[VEC_PHI1]], <vscale x 4 x float> %[[SEL1]])

; CHECK-UNORDERED-LABEL: @fadd_strict_interleave
; CHECK-UNORDERED: entry
; CHECK-UNORDERED: %[[ARRAYIDX:.*]] = getelementptr inbounds float, float* %a, i64 1
; CHECK-UNORDERED: %[[LOAD1:.*]] = load float, float* %a
; CHECK-UNORDERED: %[[LOAD2:.*]] = load float, float* %[[ARRAYIDX]]
; CHECK-UNORDERED: vector.ph
; CHECK-UNORDERED: %[[INS_ELT2:.*]] = insertelement <vscale x 4 x float> shufflevector (<vscale x 4 x float> insertelement (<vscale x 4 x float> poison, float -0.000000e+00, i32 0), <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer), float %[[LOAD2]], i32 0
; CHECK-UNORDERED: %[[INS_ELT1:.*]] = insertelement <vscale x 4 x float> shufflevector (<vscale x 4 x float> insertelement (<vscale x 4 x float> poison, float -0.000000e+00, i32 0), <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer), float %[[LOAD1]], i32 0
; CHECK-UNORDERED: %[[STEPVEC1:.*]] = call <vscale x 4 x i64> @llvm.experimental.stepvector.nxv4i64()
; CHECK-UNORDERED: %[[STEPVEC_ADD1:.*]] = add <vscale x 4 x i64> %[[STEPVEC1]], zeroinitializer
; CHECK-UNORDERED: %[[STEPVEC_MUL:.*]] = mul <vscale x 4 x i64> %[[STEPVEC_ADD1]], shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> poison, i64 2, i32 0), <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-UNORDERED: %[[INDUCTION:.*]] = add <vscale x 4 x i64> zeroinitializer, %[[STEPVEC_MUL]]
; CHECK-UNORDERED: vector.body
; CHECK-UNORDERED: %[[VEC_PHI2:.*]] = phi <vscale x 4 x float> [ %[[INS_ELT2]], %vector.ph ], [ %[[VEC_FADD2:.*]], %vector.body ]
; CHECK-UNORDERED: %[[VEC_PHI1:.*]] = phi <vscale x 4 x float> [ %[[INS_ELT1]], %vector.ph ], [ %[[VEC_FADD1:.*]], %vector.body ]
; CHECK-UNORDERED: %[[GEP1:.*]] = getelementptr inbounds float, float* %b, <vscale x 4 x i64>
; CHECK-UNORDERED: %[[MGATHER1:.*]] = call <vscale x 4 x float> @llvm.masked.gather.nxv4f32.nxv4p0f32(<vscale x 4 x float*> %[[GEP1]], i32 4, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i32 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> poison)
; CHECK-UNORDERED:  %[[VEC_FADD1]] = fadd <vscale x 4 x float> %[[MGATHER1]], %[[VEC_PHI1]]
; CHECK-UNORDERED: %[[OR:.*]] = or <vscale x 4 x i64> {{.*}}, shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> poison, i64 1, i32 0), <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-UNORDERED: %[[GEP2:.*]] = getelementptr inbounds float, float* %b, <vscale x 4 x i64> %[[OR]]
; CHECK-UNORDERED: %[[MGATHER2:.*]] = call <vscale x 4 x float> @llvm.masked.gather.nxv4f32.nxv4p0f32(<vscale x 4 x float*> %[[GEP2]], i32 4, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i32 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> poison)
; CHECK-UNORDERED: %[[VEC_FADD2]] = fadd <vscale x 4 x float> %[[MGATHER2]], %[[VEC_PHI2]]
; CHECK-UNORDERED-NOT: call float @llvm.vector.reduce.fadd
; CHECK-UNORDERED: middle.block
; CHECK-UNORDERED: %[[VEC_RDX1:.*]] = call float @llvm.vector.reduce.fadd.nxv4f32(float -0.000000e+00, <vscale x 4 x float> %[[VEC_FADD1]])
; CHECK-UNORDERED: %[[VEC_RDX2:.*]] = call float @llvm.vector.reduce.fadd.nxv4f32(float -0.000000e+00, <vscale x 4 x float> %[[VEC_FADD2]])
; CHECK-UNORDERED: for.body
; CHECK-UNORDERED: %[[LOAD3:.*]] = load float, float*
; CHECK-UNORDERED: %[[FADD1:.*]] = fadd float %[[LOAD3]], {{.*}}
; CHECK-UNORDERED: %[[LOAD4:.*]] = load float, float*
; CHECK-UNORDERED: %[[FADD2:.*]] = fadd float %[[LOAD4]], {{.*}}
; CHECK-UNORDERED: for.end
; CHECK-UNORDERED: %[[RDX1:.*]] = phi float [ %[[FADD1]], %for.body ], [ %[[VEC_RDX1]], %middle.block ]
; CHECK-UNORDERED: %[[RDX2:.*]] = phi float [ %[[FADD2]], %for.body ], [ %[[VEC_RDX2]], %middle.block ]
; CHECK-UNORDERED: store float %[[RDX1]], float* %a
; CHECK-UNORDERED: store float %[[RDX2]], float* {{.*}}
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

define float @fadd_of_sum(float* noalias nocapture readonly %a, float* noalias nocapture readonly %b, i64 %n) #0 {
; CHECK-ORDERED-LABEL: @fadd_of_sum
; CHECK-ORDERED: vector.body
; CHECK-ORDERED: %[[VEC_PHI1:.*]] = phi float [ 0.000000e+00, %vector.ph ], [ %[[RDX:.*]], %vector.body ]
; CHECK-ORDERED: %[[LOAD1:.*]] = load <vscale x 4 x float>, <vscale x 4 x float>*
; CHECK-ORDERED: %[[LOAD2:.*]] = load <vscale x 4 x float>, <vscale x 4 x float>*
; CHECK-ORDERED: %[[ADD:.*]] = fadd <vscale x 4 x float> %[[LOAD1]], %[[LOAD2]]
; CHECK-ORDERED: %[[RDX]] = call float @llvm.vector.reduce.fadd.nxv4f32(float %[[VEC_PHI1]], <vscale x 4 x float> %[[ADD]])
; CHECK-ORDERED: for.end.loopexit
; CHECK-ORDERED: %[[EXIT_PHI:.*]] = phi float [ {{.*}}, %for.body ], [ %[[RDX]], %middle.block ]
; CHECK-ORDERED: for.end
; CHECK-ORDERED: %[[PHI:.*]] = phi float [ 0.000000e+00, %entry ], [ %[[EXIT_PHI]], %for.end.loopexit ]
; CHECK-ORDERED: ret float %[[PHI]]

; CHECK-ORDERED-TF-LABEL: @fadd_of_sum
; CHECK-ORDERED-TF: vector.body
; CHECK-ORDERED-TF: %[[ACTIVE_LANE_MASK:.*]] = phi <vscale x 4 x i1>
; CHECK-ORDERED-TF: %[[VEC_PHI1:.*]] = phi float [ 0.000000e+00, %vector.ph ], [ %[[RDX:.*]], %vector.body ]
; CHECK-ORDERED-TF: %[[LOAD1:.*]] = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0nxv4f32(<vscale x 4 x float>*
; CHECK-ORDERED-TF: %[[LOAD2:.*]] = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0nxv4f32(<vscale x 4 x float>*
; CHECK-ORDERED-TF: %[[ADD:.*]] = fadd <vscale x 4 x float> %[[LOAD1]], %[[LOAD2]]
; CHECK-ORDERED-TF: %[[SEL:.*]] = select <vscale x 4 x i1> %[[ACTIVE_LANE_MASK]], <vscale x 4 x float> %[[ADD]], <vscale x 4 x float> shufflevector (<vscale x 4 x float> insertelement (<vscale x 4 x float> poison, float -0.000000e+00, i32 0), <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-ORDERED-TF: %[[RDX]] = call float @llvm.vector.reduce.fadd.nxv4f32(float %[[VEC_PHI1]], <vscale x 4 x float> %[[SEL]])
; CHECK-ORDERED-TF: for.end.loopexit
; CHECK-ORDERED-TF: %[[EXIT_PHI:.*]] = phi float [ {{.*}}, %for.body ], [ %[[RDX]], %middle.block ]
; CHECK-ORDERED-TF: for.end
; CHECK-ORDERED-TF: %[[PHI:.*]] = phi float [ 0.000000e+00, %entry ], [ %[[EXIT_PHI]], %for.end.loopexit ]
; CHECK-ORDERED-TF: ret float %[[PHI]]

; CHECK-UNORDERED-LABEL: @fadd_of_sum
; CHECK-UNORDERED: vector.body
; CHECK-UNORDERED: %[[VEC_PHI:.*]] = phi <vscale x 4 x float> [ insertelement (<vscale x 4 x float> shufflevector (<vscale x 4 x float> insertelement (<vscale x 4 x float> poison, float -0.000000e+00, i32 0), <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer), float 0.000000e+00, i32 0), %vector.ph ], [ %[[VEC_FADD2:.*]], %vector.body ]
; CHECK-UNORDERED: %[[VEC_LOAD1:.*]] = load <vscale x 4 x float>, <vscale x 4 x float>*
; CHECK-UNORDERED: %[[VEC_LOAD2:.*]] = load <vscale x 4 x float>, <vscale x 4 x float>*
; CHECK-UNORDERED: %[[VEC_FADD1:.*]] = fadd <vscale x 4 x float> %[[VEC_LOAD1]], %[[VEC_LOAD2]]
; CHECK-UNORDERED: %[[VEC_FADD2]] = fadd <vscale x 4 x float> %[[VEC_PHI]], %[[VEC_FADD1]]
; CHECK-UNORDERED-NOT: call float @llvm.vector.reduce.fadd
; CHECK-UNORDERED: middle.block
; CHECK-UNORDERED: %[[RDX:.*]] = call float @llvm.vector.reduce.fadd.nxv4f32(float -0.000000e+00, <vscale x 4 x float> %[[VEC_FADD2]])
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

define float @fadd_conditional(float* noalias nocapture readonly %a, float* noalias nocapture readonly %b, i64 %n) #0 {
; CHECK-ORDERED-LABEL: @fadd_conditional
; CHECK-ORDERED: vector.body
; CHECK-ORDERED: %[[VEC_PHI:.*]] = phi float [ 1.000000e+00, %vector.ph ], [ %[[RDX:.*]], %vector.body ]
; CHECK-ORDERED: %[[LOAD:.*]] = load <vscale x 4 x float>, <vscale x 4 x float>*
; CHECK-ORDERED: %[[FCMP:.*]] = fcmp une <vscale x 4 x float> %[[LOAD]], zeroinitializer
; CHECK-ORDERED: %[[MASKED_LOAD:.*]] = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0nxv4f32(<vscale x 4 x float>* {{.*}}, i32 4, <vscale x 4 x i1> %[[FCMP]], <vscale x 4 x float> poison)
; CHECK-ORDERED: %[[XOR:.*]] = xor <vscale x 4 x i1> %[[FCMP]], shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i32 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-ORDERED: %[[SELECT:.*]] = select <vscale x 4 x i1> %[[XOR]], <vscale x 4 x float> shufflevector (<vscale x 4 x float> insertelement (<vscale x 4 x float> poison, float 3.000000e+00, i32 0), <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %[[MASKED_LOAD]]
; CHECK-ORDERED: %[[RDX]] = call float @llvm.vector.reduce.fadd.nxv4f32(float %[[VEC_PHI]], <vscale x 4 x float> %[[SELECT]])
; CHECK-ORDERED: scalar.ph
; CHECK-ORDERED: %[[MERGE_RDX:.*]] = phi float [ 1.000000e+00, %entry ], [ %[[RDX]], %middle.block ]
; CHECK-ORDERED: for.body
; CHECK-ORDERED: %[[RES:.*]] = phi float [ %[[MERGE_RDX]], %scalar.ph ], [ %[[FADD:.*]], %for.inc ]
; CHECK-ORDERED: if.then
; CHECK-ORDERED: %[[LOAD2:.*]] = load float, float*
; CHECK-ORDERED: for.inc
; CHECK-ORDERED: %[[PHI:.*]] = phi float [ %[[LOAD2]], %if.then ], [ 3.000000e+00, %for.body ]
; CHECK-ORDERED: %[[FADD]] = fadd float %[[RES]], %[[PHI]]
; CHECK-ORDERED: for.end
; CHECK-ORDERED: %[[RDX_PHI:.*]] = phi float [ %[[FADD]], %for.inc ], [ %[[RDX]], %middle.block ]
; CHECK-ORDERED: ret float %[[RDX_PHI]]

; CHECK-ORDERED-TF-LABEL: @fadd_conditional
; CHECK-ORDERED-TF: vector.body
; CHECK-ORDERED-TF: %[[ACTIVE_LANE_MASK:.*]] = phi <vscale x 4 x i1>
; CHECK-ORDERED-TF: %[[VEC_PHI:.*]] = phi float [ 1.000000e+00, %vector.ph ], [ %[[RDX:.*]], %vector.body ]
; CHECK-ORDERED-TF: %[[LOAD:.*]] = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0nxv4f32(<vscale x 4 x float>*
; CHECK-ORDERED-TF: %[[FCMP:.*]] = fcmp une <vscale x 4 x float> %[[LOAD]], zeroinitializer
; CHECK-ORDERED-TF: %[[SELECT0:.*]] = select <vscale x 4 x i1> %[[ACTIVE_LANE_MASK]], <vscale x 4 x i1> %[[FCMP]], <vscale x 4 x i1> zeroinitializer
; CHECK-ORDERED-TF: %[[MASKED_LOAD:.*]] = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0nxv4f32(<vscale x 4 x float>* {{.*}}, i32 4, <vscale x 4 x i1> %[[SELECT0]], <vscale x 4 x float> poison)
; CHECK-ORDERED-TF: %[[XOR:.*]] = xor <vscale x 4 x i1> %[[FCMP]], shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i32 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-ORDERED-TF: %[[SELECT1:.*]] = select <vscale x 4 x i1> %[[ACTIVE_LANE_MASK]], <vscale x 4 x i1> %[[XOR]], <vscale x 4 x i1> zeroinitializer
; CHECK-ORDERED-TF: %[[SELECT2:.*]] = select <vscale x 4 x i1> %[[SELECT1]], <vscale x 4 x float> shufflevector (<vscale x 4 x float> insertelement (<vscale x 4 x float> poison, float 3.000000e+00, i32 0), <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %[[MASKED_LOAD]]
; CHECK-ORDERED-TF: %[[OR:.*]] = or <vscale x 4 x i1> %[[SELECT0]], %[[SELECT1]]
; CHECK-ORDERED-TF: %[[SELECT3:.*]] = select <vscale x 4 x i1> %[[OR]], <vscale x 4 x float> %[[SELECT2]], <vscale x 4 x float> shufflevector (<vscale x 4 x float> insertelement (<vscale x 4 x float> poison, float -0.000000e+00, i32 0), <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-ORDERED-TF: %[[RDX]] = call float @llvm.vector.reduce.fadd.nxv4f32(float %[[VEC_PHI]], <vscale x 4 x float> %[[SELECT3]])
; CHECK-ORDERED-TF: scalar.ph
; CHECK-ORDERED-TF: %[[MERGE_RDX:.*]] = phi float [ 1.000000e+00, %entry ], [ %[[RDX]], %middle.block ]
; CHECK-ORDERED-TF: for.body
; CHECK-ORDERED-TF: %[[RES:.*]] = phi float [ %[[MERGE_RDX]], %scalar.ph ], [ %[[FADD:.*]], %for.inc ]
; CHECK-ORDERED-TF: if.then
; CHECK-ORDERED-TF: %[[LOAD2:.*]] = load float, float*
; CHECK-ORDERED-TF: for.inc
; CHECK-ORDERED-TF: %[[PHI:.*]] = phi float [ %[[LOAD2]], %if.then ], [ 3.000000e+00, %for.body ]
; CHECK-ORDERED-TF: %[[FADD]] = fadd float %[[RES]], %[[PHI]]
; CHECK-ORDERED-TF: for.end
; CHECK-ORDERED-TF: %[[RDX_PHI:.*]] = phi float [ %[[FADD]], %for.inc ], [ %[[RDX]], %middle.block ]
; CHECK-ORDERED-TF: ret float %[[RDX_PHI]]

; CHECK-UNORDERED-LABEL: @fadd_conditional
; CHECK-UNORDERED: vector.body
; CHECK-UNORDERED: %[[VEC_PHI:.*]] = phi <vscale x 4 x float> [ insertelement (<vscale x 4 x float> shufflevector (<vscale x 4 x float> insertelement (<vscale x 4 x float> poison, float -0.000000e+00, i32 0), <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer), float 1.000000e+00, i32 0), %vector.ph ], [ %[[VEC_FADD:.*]], %vector.body ]
; CHECK-UNORDERED: %[[LOAD1:.*]] = load <vscale x 4 x float>, <vscale x 4 x float>*
; CHECK-UNORDERED: %[[FCMP:.*]] = fcmp une <vscale x 4 x float> %[[LOAD1]], zeroinitializer
; CHECK-UNORDERED: %[[MASKED_LOAD:.*]] = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0nxv4f32(<vscale x 4 x float>* {{.*}}, i32 4, <vscale x 4 x i1> %[[FCMP]], <vscale x 4 x float> poison)
; CHECK-UNORDERED: %[[XOR:.*]] = xor <vscale x 4 x i1> %[[FCMP]], shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i32 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-UNORDERED: %[[SELECT:.*]] = select <vscale x 4 x i1> %[[XOR]], <vscale x 4 x float> shufflevector (<vscale x 4 x float> insertelement (<vscale x 4 x float> poison, float 3.000000e+00, i32 0), <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %[[MASKED_LOAD]]
; CHECK-UNORDERED: %[[VEC_FADD]] = fadd <vscale x 4 x float> %[[VEC_PHI]], %[[SELECT]]
; CHECK-UNORDERED-NOT: call float @llvm.vector.reduce.fadd
; CHECK-UNORDERED: middle.block
; CHECK-UNORDERED: %[[RDX:.*]] = call float @llvm.vector.reduce.fadd.nxv4f32(float -0.000000e+00, <vscale x 4 x float> %[[VEC_FADD]])
; CHECK-UNORDERED: for.body
; CHECK-UNORDERED: %[[RES:.*]] = phi float [ %bc.merge.rdx, %scalar.ph ], [ %[[FADD:.*]], %for.inc ]
; CHECK-UNORDERED: for.inc
; CHECK-UNORDERED: %[[FADD]] = fadd float %[[RES]], {{.*}}
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

; Negative test - loop contains multiple fadds which we cannot safely reorder
define float @fadd_multiple(float* noalias nocapture %a, float* noalias nocapture %b, i64 %n) #0 {
; CHECK-ORDERED-LABEL: @fadd_multiple
; CHECK-ORDERED-NOT: vector.body

; CHECK-ORDERED-TF-LABEL: @fadd_multiple
; CHECK-ORDERED-TF-NOT: vector.body

; CHECK-UNORDERED-LABEL: @fadd_multiple
; CHECK-UNORDERED: vector.body
; CHECK-UNORDERED: %[[PHI:.*]] = phi <vscale x 8 x float> [ insertelement (<vscale x 8 x float> shufflevector (<vscale x 8 x float> insertelement (<vscale x 8 x float> poison, float -0.000000e+00, i32 0), <vscale x 8 x float> poison, <vscale x 8 x i32> zeroinitializer), float -0.000000e+00, i32 0), %vector.ph ], [ %[[VEC_FADD2:.*]], %vector.body ]
; CHECK-UNORDERED: %[[VEC_LOAD1:.*]] = load <vscale x 8 x float>, <vscale x 8 x float>
; CHECK-UNORDERED: %[[VEC_FADD1:.*]] = fadd <vscale x 8 x float> %[[PHI]], %[[VEC_LOAD1]]
; CHECK-UNORDERED: %[[VEC_LOAD2:.*]] = load <vscale x 8 x float>, <vscale x 8 x float>
; CHECK-UNORDERED: %[[VEC_FADD2]] = fadd <vscale x 8 x float> %[[VEC_FADD1]], %[[VEC_LOAD2]]
; CHECK-UNORDERED: middle.block
; CHECK-UNORDERED: %[[RDX:.*]] = call float @llvm.vector.reduce.fadd.nxv8f32(float -0.000000e+00, <vscale x 8 x float> %[[VEC_FADD2]])
; CHECK-UNORDERED: for.body
; CHECK-UNORDERED: %[[SUM:.*]] = phi float [ %bc.merge.rdx, %scalar.ph ], [ %[[FADD2:.*]], %for.body ]
; CHECK-UNORDERED: %[[LOAD1:.*]] = load float, float*
; CHECK-UNORDERED: %[[FADD1:.*]] = fadd float %[[SUM]], %[[LOAD1]]
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

; Test case where loop has a call to the llvm.fmuladd intrinsic.
define float @fmuladd_strict(float* %a, float* %b, i64 %n) #0 {
; CHECK-ORDERED-LABEL: @fmuladd_strict
; CHECK-ORDERED: vector.body:
; CHECK-ORDERED: [[VEC_PHI:%.*]] = phi float [ 0.000000e+00, %vector.ph ], [ [[RDX3:%.*]], %vector.body ]
; CHECK-ORDERED: [[WIDE_LOAD:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-ORDERED: [[WIDE_LOAD1:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-ORDERED: [[WIDE_LOAD2:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-ORDERED: [[WIDE_LOAD3:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-ORDERED: [[WIDE_LOAD4:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-ORDERED: [[WIDE_LOAD5:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-ORDERED: [[WIDE_LOAD6:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-ORDERED: [[WIDE_LOAD7:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-ORDERED: [[FMUL:%.*]] = fmul <vscale x 8 x float> [[WIDE_LOAD]], [[WIDE_LOAD4]]
; CHECK-ORDERED: [[FMUL1:%.*]] = fmul <vscale x 8 x float> [[WIDE_LOAD1]], [[WIDE_LOAD5]]
; CHECK-ORDERED: [[FMUL2:%.*]] = fmul <vscale x 8 x float> [[WIDE_LOAD2]], [[WIDE_LOAD6]]
; CHECK-ORDERED: [[FMUL3:%.*]] = fmul <vscale x 8 x float> [[WIDE_LOAD3]], [[WIDE_LOAD7]]
; CHECK-ORDERED: [[RDX:%.*]] = call float @llvm.vector.reduce.fadd.nxv8f32(float [[VEC_PHI]], <vscale x 8 x float> [[FMUL]])
; CHECK-ORDERED: [[RDX1:%.*]] = call float @llvm.vector.reduce.fadd.nxv8f32(float [[RDX]], <vscale x 8 x float> [[FMUL1]])
; CHECK-ORDERED: [[RDX2:%.*]] = call float @llvm.vector.reduce.fadd.nxv8f32(float [[RDX1]], <vscale x 8 x float> [[FMUL2]])
; CHECK-ORDERED: [[RDX3]] = call float @llvm.vector.reduce.fadd.nxv8f32(float [[RDX2]], <vscale x 8 x float> [[FMUL3]])
; CHECK-ORDERED: for.end
; CHECK-ORDERED: [[RES:%.*]] = phi float [ [[SCALAR:%.*]], %for.body ], [ [[RDX3]], %middle.block ]
; CHECK-ORDERED: ret float [[RES]]

; CHECK-ORDERED-TF-LABEL: @fmuladd_strict
; CHECK-ORDERED-TF: vector.body:
; CHECK-ORDERED-TF: [[ACTIVE_LANE_MASK:%.*]] = phi <vscale x 8 x i1>
; CHECK-ORDERED-TF: [[ACTIVE_LANE_MASK1:%.*]] = phi <vscale x 8 x i1>
; CHECK-ORDERED-TF: [[ACTIVE_LANE_MASK2:%.*]] = phi <vscale x 8 x i1>
; CHECK-ORDERED-TF: [[ACTIVE_LANE_MASK3:%.*]] = phi <vscale x 8 x i1>
; CHECK-ORDERED-TF: [[VEC_PHI:%.*]] = phi float [ 0.000000e+00, %vector.ph ], [ [[RDX3:%.*]], %vector.body ]
; CHECK-ORDERED-TF: [[WIDE_LOAD:%.*]] = call <vscale x 8 x float> @llvm.masked.load.nxv8f32
; CHECK-ORDERED-TF: [[WIDE_LOAD1:%.*]] = call <vscale x 8 x float> @llvm.masked.load.nxv8f32
; CHECK-ORDERED-TF: [[WIDE_LOAD2:%.*]] = call <vscale x 8 x float> @llvm.masked.load.nxv8f32
; CHECK-ORDERED-TF: [[WIDE_LOAD3:%.*]] = call <vscale x 8 x float> @llvm.masked.load.nxv8f32
; CHECK-ORDERED-TF: [[WIDE_LOAD4:%.*]] = call <vscale x 8 x float> @llvm.masked.load.nxv8f32
; CHECK-ORDERED-TF: [[WIDE_LOAD5:%.*]] = call <vscale x 8 x float> @llvm.masked.load.nxv8f32
; CHECK-ORDERED-TF: [[WIDE_LOAD6:%.*]] = call <vscale x 8 x float> @llvm.masked.load.nxv8f32
; CHECK-ORDERED-TF: [[WIDE_LOAD7:%.*]] = call <vscale x 8 x float> @llvm.masked.load.nxv8f32
; CHECK-ORDERED-TF: [[FMUL:%.*]] = fmul <vscale x 8 x float> [[WIDE_LOAD]], [[WIDE_LOAD4]]
; CHECK-ORDERED-TF: [[FMUL1:%.*]] = fmul <vscale x 8 x float> [[WIDE_LOAD1]], [[WIDE_LOAD5]]
; CHECK-ORDERED-TF: [[FMUL2:%.*]] = fmul <vscale x 8 x float> [[WIDE_LOAD2]], [[WIDE_LOAD6]]
; CHECK-ORDERED-TF: [[FMUL3:%.*]] = fmul <vscale x 8 x float> [[WIDE_LOAD3]], [[WIDE_LOAD7]]
; CHECK-ORDERED-TF: [[SEL:%.*]] = select <vscale x 8 x i1> [[ACTIVE_LANE_MASK]], <vscale x 8 x float> [[FMUL]], <vscale x 8 x float> shufflevector (<vscale x 8 x float> insertelement (<vscale x 8 x float> poison, float -0.000000e+00, i32 0), <vscale x 8 x float> poison, <vscale x 8 x i32> zeroinitializer)
; CHECK-ORDERED-TF: [[RDX:%.*]] = call float @llvm.vector.reduce.fadd.nxv8f32(float [[VEC_PHI]], <vscale x 8 x float> [[SEL]])
; CHECK-ORDERED-TF: [[SEL1:%.*]] = select <vscale x 8 x i1> [[ACTIVE_LANE_MASK1]], <vscale x 8 x float> [[FMUL1]], <vscale x 8 x float> shufflevector (<vscale x 8 x float> insertelement (<vscale x 8 x float> poison, float -0.000000e+00, i32 0), <vscale x 8 x float> poison, <vscale x 8 x i32> zeroinitializer)
; CHECK-ORDERED-TF: [[RDX1:%.*]] = call float @llvm.vector.reduce.fadd.nxv8f32(float [[RDX]], <vscale x 8 x float> [[SEL1]])
; CHECK-ORDERED-TF: [[SEL2:%.*]] = select <vscale x 8 x i1> [[ACTIVE_LANE_MASK2]], <vscale x 8 x float> [[FMUL2]], <vscale x 8 x float> shufflevector (<vscale x 8 x float> insertelement (<vscale x 8 x float> poison, float -0.000000e+00, i32 0), <vscale x 8 x float> poison, <vscale x 8 x i32> zeroinitializer)
; CHECK-ORDERED-TF: [[RDX2:%.*]] = call float @llvm.vector.reduce.fadd.nxv8f32(float [[RDX1]], <vscale x 8 x float> [[SEL2]])
; CHECK-ORDERED-TF: [[SEL3:%.*]] = select <vscale x 8 x i1> [[ACTIVE_LANE_MASK3]], <vscale x 8 x float> [[FMUL3]], <vscale x 8 x float> shufflevector (<vscale x 8 x float> insertelement (<vscale x 8 x float> poison, float -0.000000e+00, i32 0), <vscale x 8 x float> poison, <vscale x 8 x i32> zeroinitializer)
; CHECK-ORDERED-TF: [[RDX3]] = call float @llvm.vector.reduce.fadd.nxv8f32(float [[RDX2]], <vscale x 8 x float> [[SEL3]])
; CHECK-ORDERED-TF: for.end
; CHECK-ORDERED-TF: [[RES:%.*]] = phi float [ [[SCALAR:%.*]], %for.body ], [ [[RDX3]], %middle.block ]
; CHECK-ORDERED-TF: ret float [[RES]]

; CHECK-UNORDERED-LABEL: @fmuladd_strict
; CHECK-UNORDERED: vector.body
; CHECK-UNORDERED:    [[VEC_PHI:%.*]] = phi <vscale x 8 x float> [ insertelement (<vscale x 8 x float> shufflevector (<vscale x 8 x float> insertelement (<vscale x 8 x float> poison, float -0.000000e+00, i32 0), <vscale x 8 x float> poison, <vscale x 8 x i32> zeroinitializer), float 0.000000e+00, i32 0), %vector.ph ], [ [[FMULADD:%.*]], %vector.body ]
; CHECK-UNORDERED:    [[VEC_PHI1:%.*]] = phi <vscale x 8 x float> [ shufflevector (<vscale x 8 x float> insertelement (<vscale x 8 x float> poison, float -0.000000e+00, i32 0), <vscale x 8 x float> poison, <vscale x 8 x i32> zeroinitializer), %vector.ph ], [ [[FMULADD1:%.*]], %vector.body ]
; CHECK-UNORDERED:    [[VEC_PHI2:%.*]] = phi <vscale x 8 x float> [ shufflevector (<vscale x 8 x float> insertelement (<vscale x 8 x float> poison, float -0.000000e+00, i32 0), <vscale x 8 x float> poison, <vscale x 8 x i32> zeroinitializer), %vector.ph ], [ [[FMULADD2:%.*]], %vector.body ]
; CHECK-UNORDERED:    [[VEC_PHI3:%.*]] = phi <vscale x 8 x float> [ shufflevector (<vscale x 8 x float> insertelement (<vscale x 8 x float> poison, float -0.000000e+00, i32 0), <vscale x 8 x float> poison, <vscale x 8 x i32> zeroinitializer), %vector.ph ], [ [[FMULADD3:%.*]], %vector.body ]
; CHECK-UNORDERED: [[WIDE_LOAD:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-UNORDERED: [[WIDE_LOAD1:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-UNORDERED: [[WIDE_LOAD2:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-UNORDERED: [[WIDE_LOAD3:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-UNORDERED: [[WIDE_LOAD4:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-UNORDERED: [[WIDE_LOAD5:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-UNORDERED: [[WIDE_LOAD6:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-UNORDERED: [[WIDE_LOAD7:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-UNORDERED: [[FMULADD]] = call <vscale x 8 x float> @llvm.fmuladd.nxv8f32(<vscale x 8 x float> [[WIDE_LOAD]], <vscale x 8 x float> [[WIDE_LOAD4]], <vscale x 8 x float> [[VEC_PHI]])
; CHECK-UNORDERED: [[FMULADD1]] = call <vscale x 8 x float> @llvm.fmuladd.nxv8f32(<vscale x 8 x float> [[WIDE_LOAD1]], <vscale x 8 x float> [[WIDE_LOAD5]], <vscale x 8 x float> [[VEC_PHI1]])
; CHECK-UNORDERED: [[FMULADD2]] = call <vscale x 8 x float> @llvm.fmuladd.nxv8f32(<vscale x 8 x float> [[WIDE_LOAD2]], <vscale x 8 x float> [[WIDE_LOAD6]], <vscale x 8 x float> [[VEC_PHI2]])
; CHECK-UNORDERED: [[FMULADD3]] = call <vscale x 8 x float> @llvm.fmuladd.nxv8f32(<vscale x 8 x float> [[WIDE_LOAD3]], <vscale x 8 x float> [[WIDE_LOAD7]], <vscale x 8 x float> [[VEC_PHI3]])
; CHECK-UNORDERED-NOT: llvm.vector.reduce.fadd
; CHECK-UNORDERED: middle.block
; CHECK-UNORDERED: [[BIN_RDX:%.*]] = fadd <vscale x 8 x float> [[FMULADD1]], [[FMULADD]]
; CHECK-UNORDERED: [[BIN_RDX1:%.*]] = fadd <vscale x 8 x float> [[FMULADD2]], [[BIN_RDX]]
; CHECK-UNORDERED: [[BIN_RDX2:%.*]] = fadd <vscale x 8 x float> [[FMULADD3]], [[BIN_RDX1]]
; CHECK-UNORDERED: [[RDX:%.*]] = call float @llvm.vector.reduce.fadd.nxv8f32(float -0.000000e+00, <vscale x 8 x float> [[BIN_RDX2]]
; CHECK-UNORDERED: for.body
; CHECK-UNORDERED: [[SUM_07:%.*]] = phi float [ [[SCALAR:%.*]], %scalar.ph ], [ [[MULADD:%.*]], %for.body ]
; CHECK-UNORDERED: [[LOAD:%.*]] = load float, float*
; CHECK-UNORDERED: [[LOAD1:%.*]] = load float, float*
; CHECK-UNORDERED: [[MULADD]] = tail call float @llvm.fmuladd.f32(float [[LOAD]], float [[LOAD1]], float [[SUM_07]])
; CHECK-UNORDERED: for.end
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

; Same as above but where the call to the llvm.fmuladd intrinsic has a fast-math flag.
define float @fmuladd_strict_fmf(float* %a, float* %b, i64 %n) #0 {
; CHECK-ORDERED-LABEL: @fmuladd_strict_fmf
; CHECK-ORDERED: vector.body:
; CHECK-ORDERED: [[VEC_PHI:%.*]] = phi float [ 0.000000e+00, %vector.ph ], [ [[RDX3:%.*]], %vector.body ]
; CHECK-ORDERED: [[WIDE_LOAD:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-ORDERED: [[WIDE_LOAD1:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-ORDERED: [[WIDE_LOAD2:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-ORDERED: [[WIDE_LOAD3:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-ORDERED: [[WIDE_LOAD4:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-ORDERED: [[WIDE_LOAD5:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-ORDERED: [[WIDE_LOAD6:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-ORDERED: [[WIDE_LOAD7:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-ORDERED: [[FMUL:%.*]] = fmul nnan <vscale x 8 x float> [[WIDE_LOAD]], [[WIDE_LOAD4]]
; CHECK-ORDERED: [[FMUL1:%.*]] = fmul nnan <vscale x 8 x float> [[WIDE_LOAD1]], [[WIDE_LOAD5]]
; CHECK-ORDERED: [[FMUL2:%.*]] = fmul nnan <vscale x 8 x float> [[WIDE_LOAD2]], [[WIDE_LOAD6]]
; CHECK-ORDERED: [[FMUL3:%.*]] = fmul nnan <vscale x 8 x float> [[WIDE_LOAD3]], [[WIDE_LOAD7]]
; CHECK-ORDERED: [[RDX:%.*]] = call nnan float @llvm.vector.reduce.fadd.nxv8f32(float [[VEC_PHI]], <vscale x 8 x float> [[FMUL]])
; CHECK-ORDERED: [[RDX1:%.*]] = call nnan float @llvm.vector.reduce.fadd.nxv8f32(float [[RDX]], <vscale x 8 x float> [[FMUL1]])
; CHECK-ORDERED: [[RDX2:%.*]] = call nnan float @llvm.vector.reduce.fadd.nxv8f32(float [[RDX1]], <vscale x 8 x float> [[FMUL2]])
; CHECK-ORDERED: [[RDX3]] = call nnan float @llvm.vector.reduce.fadd.nxv8f32(float [[RDX2]], <vscale x 8 x float> [[FMUL3]])
; CHECK-ORDERED: for.end
; CHECK-ORDERED: [[RES:%.*]] = phi float [ [[SCALAR:%.*]], %for.body ], [ [[RDX3]], %middle.block ]
; CHECK-ORDERED: ret float [[RES]]

; CHECK-ORDERED-TF-LABEL: @fmuladd_strict_fmf
; CHECK-ORDERED-TF: vector.body:
; CHECK-ORDERED-TF: [[ACTIVE_LANE_MASK:%.*]] = phi <vscale x 8 x i1>
; CHECK-ORDERED-TF: [[ACTIVE_LANE_MASK1:%.*]] = phi <vscale x 8 x i1>
; CHECK-ORDERED-TF: [[ACTIVE_LANE_MASK2:%.*]] = phi <vscale x 8 x i1>
; CHECK-ORDERED-TF: [[ACTIVE_LANE_MASK3:%.*]] = phi <vscale x 8 x i1>
; CHECK-ORDERED-TF: [[VEC_PHI:%.*]] = phi float [ 0.000000e+00, %vector.ph ], [ [[RDX3:%.*]], %vector.body ]
; CHECK-ORDERED-TF: [[WIDE_LOAD:%.*]] = call <vscale x 8 x float> @llvm.masked.load.nxv8f32
; CHECK-ORDERED-TF: [[WIDE_LOAD1:%.*]] = call <vscale x 8 x float> @llvm.masked.load.nxv8f32
; CHECK-ORDERED-TF: [[WIDE_LOAD2:%.*]] = call <vscale x 8 x float> @llvm.masked.load.nxv8f32
; CHECK-ORDERED-TF: [[WIDE_LOAD3:%.*]] = call <vscale x 8 x float> @llvm.masked.load.nxv8f32
; CHECK-ORDERED-TF: [[WIDE_LOAD4:%.*]] = call <vscale x 8 x float> @llvm.masked.load.nxv8f32
; CHECK-ORDERED-TF: [[WIDE_LOAD5:%.*]] = call <vscale x 8 x float> @llvm.masked.load.nxv8f32
; CHECK-ORDERED-TF: [[WIDE_LOAD6:%.*]] = call <vscale x 8 x float> @llvm.masked.load.nxv8f32
; CHECK-ORDERED-TF: [[WIDE_LOAD7:%.*]] = call <vscale x 8 x float> @llvm.masked.load.nxv8f32
; CHECK-ORDERED-TF: [[FMUL:%.*]] = fmul nnan <vscale x 8 x float> [[WIDE_LOAD]], [[WIDE_LOAD4]]
; CHECK-ORDERED-TF: [[FMUL1:%.*]] = fmul nnan <vscale x 8 x float> [[WIDE_LOAD1]], [[WIDE_LOAD5]]
; CHECK-ORDERED-TF: [[FMUL2:%.*]] = fmul nnan <vscale x 8 x float> [[WIDE_LOAD2]], [[WIDE_LOAD6]]
; CHECK-ORDERED-TF: [[FMUL3:%.*]] = fmul nnan <vscale x 8 x float> [[WIDE_LOAD3]], [[WIDE_LOAD7]]
; CHECK-ORDERED-TF: [[SEL:%.*]] = select nnan <vscale x 8 x i1> [[ACTIVE_LANE_MASK]], <vscale x 8 x float> [[FMUL]], <vscale x 8 x float> shufflevector (<vscale x 8 x float> insertelement (<vscale x 8 x float> poison, float -0.000000e+00, i32 0), <vscale x 8 x float> poison, <vscale x 8 x i32> zeroinitializer)
; CHECK-ORDERED-TF: [[RDX:%.*]] = call nnan float @llvm.vector.reduce.fadd.nxv8f32(float [[VEC_PHI]], <vscale x 8 x float> [[SEL]])
; CHECK-ORDERED-TF: [[SEL1:%.*]] = select nnan <vscale x 8 x i1> [[ACTIVE_LANE_MASK1]], <vscale x 8 x float> [[FMUL1]], <vscale x 8 x float> shufflevector (<vscale x 8 x float> insertelement (<vscale x 8 x float> poison, float -0.000000e+00, i32 0), <vscale x 8 x float> poison, <vscale x 8 x i32> zeroinitializer)
; CHECK-ORDERED-TF: [[RDX1:%.*]] = call nnan float @llvm.vector.reduce.fadd.nxv8f32(float [[RDX]], <vscale x 8 x float> [[SEL1]])
; CHECK-ORDERED-TF: [[SEL2:%.*]] = select nnan <vscale x 8 x i1> [[ACTIVE_LANE_MASK2]], <vscale x 8 x float> [[FMUL2]], <vscale x 8 x float> shufflevector (<vscale x 8 x float> insertelement (<vscale x 8 x float> poison, float -0.000000e+00, i32 0), <vscale x 8 x float> poison, <vscale x 8 x i32> zeroinitializer)
; CHECK-ORDERED-TF: [[RDX2:%.*]] = call nnan float @llvm.vector.reduce.fadd.nxv8f32(float [[RDX1]], <vscale x 8 x float> [[SEL2]])
; CHECK-ORDERED-TF: [[SEL3:%.*]] = select nnan <vscale x 8 x i1> [[ACTIVE_LANE_MASK3]], <vscale x 8 x float> [[FMUL3]], <vscale x 8 x float> shufflevector (<vscale x 8 x float> insertelement (<vscale x 8 x float> poison, float -0.000000e+00, i32 0), <vscale x 8 x float> poison, <vscale x 8 x i32> zeroinitializer)
; CHECK-ORDERED-TF: [[RDX3]] = call nnan float @llvm.vector.reduce.fadd.nxv8f32(float [[RDX2]], <vscale x 8 x float> [[SEL3]])
; CHECK-ORDERED-TF: for.end
; CHECK-ORDERED-TF: [[RES:%.*]] = phi float [ [[SCALAR:%.*]], %for.body ], [ [[RDX3]], %middle.block ]
; CHECK-ORDERED-TF: ret float [[RES]]

; CHECK-UNORDERED-LABEL: @fmuladd_strict_fmf
; CHECK-UNORDERED: vector.body
; CHECK-UNORDERED:    [[VEC_PHI:%.*]] = phi <vscale x 8 x float> [ insertelement (<vscale x 8 x float> shufflevector (<vscale x 8 x float> insertelement (<vscale x 8 x float> poison, float -0.000000e+00, i32 0), <vscale x 8 x float> poison, <vscale x 8 x i32> zeroinitializer), float 0.000000e+00, i32 0), %vector.ph ], [ [[FMULADD:%.*]], %vector.body ]
; CHECK-UNORDERED:    [[VEC_PHI1:%.*]] = phi <vscale x 8 x float> [ shufflevector (<vscale x 8 x float> insertelement (<vscale x 8 x float> poison, float -0.000000e+00, i32 0), <vscale x 8 x float> poison, <vscale x 8 x i32> zeroinitializer), %vector.ph ], [ [[FMULADD1:%.*]], %vector.body ]
; CHECK-UNORDERED:    [[VEC_PHI2:%.*]] = phi <vscale x 8 x float> [ shufflevector (<vscale x 8 x float> insertelement (<vscale x 8 x float> poison, float -0.000000e+00, i32 0), <vscale x 8 x float> poison, <vscale x 8 x i32> zeroinitializer), %vector.ph ], [ [[FMULADD2:%.*]], %vector.body ]
; CHECK-UNORDERED:    [[VEC_PHI3:%.*]] = phi <vscale x 8 x float> [ shufflevector (<vscale x 8 x float> insertelement (<vscale x 8 x float> poison, float -0.000000e+00, i32 0), <vscale x 8 x float> poison, <vscale x 8 x i32> zeroinitializer), %vector.ph ], [ [[FMULADD3:%.*]], %vector.body ]
; CHECK-UNORDERED: [[WIDE_LOAD:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-UNORDERED: [[WIDE_LOAD1:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-UNORDERED: [[WIDE_LOAD2:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-UNORDERED: [[WIDE_LOAD3:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-UNORDERED: [[WIDE_LOAD4:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-UNORDERED: [[WIDE_LOAD5:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-UNORDERED: [[WIDE_LOAD6:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-UNORDERED: [[WIDE_LOAD7:%.*]] = load <vscale x 8 x float>, <vscale x 8 x float>*
; CHECK-UNORDERED: [[FMULADD]] = call nnan <vscale x 8 x float> @llvm.fmuladd.nxv8f32(<vscale x 8 x float> [[WIDE_LOAD]], <vscale x 8 x float> [[WIDE_LOAD4]], <vscale x 8 x float> [[VEC_PHI]])
; CHECK-UNORDERED: [[FMULADD1]] = call nnan <vscale x 8 x float> @llvm.fmuladd.nxv8f32(<vscale x 8 x float> [[WIDE_LOAD1]], <vscale x 8 x float> [[WIDE_LOAD5]], <vscale x 8 x float> [[VEC_PHI1]])
; CHECK-UNORDERED: [[FMULADD2]] = call nnan <vscale x 8 x float> @llvm.fmuladd.nxv8f32(<vscale x 8 x float> [[WIDE_LOAD2]], <vscale x 8 x float> [[WIDE_LOAD6]], <vscale x 8 x float> [[VEC_PHI2]])
; CHECK-UNORDERED: [[FMULADD3]] = call nnan <vscale x 8 x float> @llvm.fmuladd.nxv8f32(<vscale x 8 x float> [[WIDE_LOAD3]], <vscale x 8 x float> [[WIDE_LOAD7]], <vscale x 8 x float> [[VEC_PHI3]])
; CHECK-UNORDERED-NOT: llvm.vector.reduce.fadd
; CHECK-UNORDERED: middle.block
; CHECK-UNORDERED: [[BIN_RDX:%.*]] = fadd nnan <vscale x 8 x float> [[FMULADD1]], [[FMULADD]]
; CHECK-UNORDERED: [[BIN_RDX1:%.*]] = fadd nnan <vscale x 8 x float> [[FMULADD2]], [[BIN_RDX]]
; CHECK-UNORDERED: [[BIN_RDX2:%.*]] = fadd nnan <vscale x 8 x float> [[FMULADD3]], [[BIN_RDX1]]
; CHECK-UNORDERED: [[RDX:%.*]] = call nnan float @llvm.vector.reduce.fadd.nxv8f32(float -0.000000e+00, <vscale x 8 x float> [[BIN_RDX2]]
; CHECK-UNORDERED: for.body
; CHECK-UNORDERED: [[SUM_07:%.*]] = phi float [ [[SCALAR:%.*]], %scalar.ph ], [ [[MULADD:%.*]], %for.body ]
; CHECK-UNORDERED: [[LOAD:%.*]] = load float, float*
; CHECK-UNORDERED: [[LOAD1:%.*]] = load float, float*
; CHECK-UNORDERED: [[MULADD]] = tail call nnan float @llvm.fmuladd.f32(float [[LOAD]], float [[LOAD1]], float [[SUM_07]])
; CHECK-UNORDERED: for.end
; CHECK-UNORDERED: [[RES:%.*]] = phi float [ [[MULADD]], %for.body ], [ [[RDX]], %middle.block ]
; CHECK-UNORDERED: ret float [[RES]]

; CHECK-NOT-VECTORIZED-LABEL: @fmuladd_strict_fmf
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
  %muladd = tail call nnan float @llvm.fmuladd.f32(float %0, float %1, float %sum.07)
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !1

for.end:
  ret float %muladd
}

declare float @llvm.fmuladd.f32(float, float, float)

attributes #0 = { vscale_range(1, 16) }
!0 = distinct !{!0, !3, !6, !8}
!1 = distinct !{!1, !3, !7, !8}
!2 = distinct !{!2, !4, !6, !8}
!3 = !{!"llvm.loop.vectorize.width", i32 8}
!4 = !{!"llvm.loop.vectorize.width", i32 4}
!5 = !{!"llvm.loop.vectorize.width", i32 2}
!6 = !{!"llvm.loop.interleave.count", i32 1}
!7 = !{!"llvm.loop.interleave.count", i32 4}
!8 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
