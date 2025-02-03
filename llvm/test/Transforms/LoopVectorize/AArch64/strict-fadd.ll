; RUN: opt < %s -passes=loop-vectorize -mtriple aarch64-unknown-linux-gnu -force-ordered-reductions=false -hints-allow-reordering=false -S 2>%t | FileCheck %s --check-prefix=CHECK-NOT-VECTORIZED
; RUN: opt < %s -passes=loop-vectorize -mtriple aarch64-unknown-linux-gnu -force-ordered-reductions=false -hints-allow-reordering=true  -S 2>%t | FileCheck %s --check-prefix=CHECK-UNORDERED
; RUN: opt < %s -passes=loop-vectorize -mtriple aarch64-unknown-linux-gnu -force-ordered-reductions=true  -hints-allow-reordering=false -S 2>%t | FileCheck %s --check-prefix=CHECK-ORDERED
; RUN: opt < %s -passes=loop-vectorize -mtriple aarch64-unknown-linux-gnu -force-ordered-reductions=true  -hints-allow-reordering=true  -S 2>%t | FileCheck %s --check-prefix=CHECK-UNORDERED
; RUN: opt < %s -passes=loop-vectorize -mtriple aarch64-unknown-linux-gnu -hints-allow-reordering=false -S 2>%t | FileCheck %s --check-prefix=CHECK-ORDERED

define float @fadd_conditional(ptr noalias nocapture readonly %a, ptr noalias nocapture readonly %b, i64 %n) {
; CHECK-ORDERED-LABEL: @fadd_conditional
; CHECK-ORDERED: vector.body:
; CHECK-ORDERED: %[[PHI:.*]] = phi float [ 1.000000e+00, %vector.ph ], [ %[[RDX:.*]], %pred.load.continue6 ]
; CHECK-ORDERED: %[[LOAD1:.*]] = load <4 x float>, ptr
; CHECK-ORDERED: %[[FCMP1:.*]] = fcmp une <4 x float> %[[LOAD1]], zeroinitializer
; CHECK-ORDERED: %[[EXTRACT:.*]] = extractelement <4 x i1> %[[FCMP1]], i32 0
; CHECK-ORDERED: br i1 %[[EXTRACT]], label %pred.load.if, label %pred.load.continue
; CHECK-ORDERED: pred.load.continue6
; CHECK-ORDERED: %[[PHI1:.*]] = phi <4 x float> [ %[[PHI0:.*]], %pred.load.continue4 ], [ %[[INS_ELT:.*]], %pred.load.if5 ]
; CHECK-ORDERED: %[[XOR:.*]] =  xor <4 x i1> %[[FCMP1]], splat (i1 true)
; CHECK-ORDERED: %[[PRED:.*]] = select <4 x i1> %[[XOR]], <4 x float> splat (float 3.000000e+00), <4 x float> %[[PHI1]]
; CHECK-ORDERED: %[[RDX]] = call float @llvm.vector.reduce.fadd.v4f32(float %[[PHI]], <4 x float> %[[PRED]])
; CHECK-ORDERED: for.body
; CHECK-ORDERED: %[[RES_PHI:.*]] = phi float [ %[[MERGE_RDX:.*]], %scalar.ph ], [ %[[FADD:.*]], %for.inc ]
; CHECK-ORDERED: %[[LOAD2:.*]] = load float, ptr
; CHECK-ORDERED: %[[FCMP2:.*]] = fcmp une float %[[LOAD2]], 0.000000e+00
; CHECK-ORDERED: br i1 %[[FCMP2]], label %if.then, label %for.inc
; CHECK-ORDERED: if.then
; CHECK-ORDERED: %[[LOAD3:.*]] = load float, ptr
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
; CHECK-UNORDERED: %[[LOAD1:.*]] = load <4 x float>, ptr
; CHECK-UNORDERED: %[[FCMP1:.*]] = fcmp une <4 x float> %[[LOAD1]], zeroinitializer
; CHECK-UNORDERED: %[[EXTRACT:.*]] = extractelement <4 x i1> %[[FCMP1]], i32 0
; CHECK-UNORDERED: br i1 %[[EXTRACT]], label %pred.load.if, label %pred.load.continue
; CHECK-UNORDERED: pred.load.continue6
; CHECK-UNORDERED: %[[XOR:.*]] =  xor <4 x i1> %[[FCMP1]], splat (i1 true)
; CHECK-UNORDERED: %[[PRED:.*]] = select <4 x i1> %[[XOR]], <4 x float> splat (float 3.000000e+00), <4 x float> %[[PRED_PHI:.*]]
; CHECK-UNORDERED: %[[VEC_FADD]] = fadd <4 x float> %[[PHI]], %[[PRED]]
; CHECK-UNORDERED-NOT: call float @llvm.vector.reduce.fadd
; CHECK-UNORDERED: middle.block
; CHECK-UNORDERED: %[[RDX:.*]] = call float @llvm.vector.reduce.fadd.v4f32(float -0.000000e+00, <4 x float> %[[VEC_FADD]])
; CHECK-UNORDERED: for.body
; CHECK-UNORDERED: %[[RES_PHI:.*]] = phi float [ %[[MERGE_RDX:.*]], %scalar.ph ], [ %[[FADD:.*]], %for.inc ]
; CHECK-UNORDERED: %[[LOAD2:.*]] = load float, ptr
; CHECK-UNORDERED: %[[FCMP2:.*]] = fcmp une float %[[LOAD2]], 0.000000e+00
; CHECK-UNORDERED: br i1 %[[FCMP2]], label %if.then, label %for.inc
; CHECK-UNORDERED: if.then
; CHECK-UNORDERED: %[[LOAD3:.*]] = load float, ptr
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
  %arrayidx = getelementptr inbounds float, ptr %b, i64 %iv
  %0 = load float, ptr %arrayidx, align 4
  %tobool = fcmp une float %0, 0.000000e+00
  br i1 %tobool, label %if.then, label %for.inc

if.then:                                      ; preds = %for.body
  %arrayidx2 = getelementptr inbounds float, ptr %a, i64 %iv
  %1 = load float, ptr %arrayidx2, align 4
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
