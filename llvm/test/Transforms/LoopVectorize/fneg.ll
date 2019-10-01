; RUN: opt %s -loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -S | FileCheck %s

define void @foo(float* %a, i64 %n) {
; CHECK:       vector.body:
; CHECK:         [[WIDE_LOAD:%.*]] = load <4 x float>, <4 x float>* {{.*}}, align 4
; CHECK-NEXT:    [[TMP4:%.*]] = fneg <4 x float> [[WIDE_LOAD]]
; CHECK:         store <4 x float> [[TMP4]], <4 x float>* {{.*}}, align 4
;
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %a, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %sub = fneg float %0
  store float %sub, float* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cmp = icmp eq i64 %indvars.iv.next, %n
  br i1 %cmp, label %for.exit, label %for.body

for.exit:
  ret void
}
