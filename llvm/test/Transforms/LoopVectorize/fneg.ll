; RUN: opt %s -loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -S | FileCheck %s

define void @foo(float* %a, i64 %n) {
; CHECK:       vector.body:
; CHECK:         [[WIDE_LOAD:%.*]] = load <4 x float>, <4 x float>* {{.*}}, align 4
; CHECK-NEXT:    [[TMP4:%.*]] = extractelement <4 x float> [[WIDE_LOAD]], i32 0
; CHECK-NEXT:    [[TMP5:%.*]] = fneg float [[TMP4]]
; CHECK-NEXT:    [[TMP6:%.*]] = extractelement <4 x float> [[WIDE_LOAD]], i32 1
; CHECK-NEXT:    [[TMP7:%.*]] = fneg float [[TMP6]]
; CHECK-NEXT:    [[TMP8:%.*]] = extractelement <4 x float> [[WIDE_LOAD]], i32 2
; CHECK-NEXT:    [[TMP9:%.*]] = fneg float [[TMP8]]
; CHECK-NEXT:    [[TMP10:%.*]] = extractelement <4 x float> [[WIDE_LOAD]], i32 3
; CHECK-NEXT:    [[TMP11:%.*]] = fneg float [[TMP10]]
; CHECK-NEXT:    [[TMP12:%.*]] = insertelement <4 x float> undef, float [[TMP5]], i32 0
; CHECK-NEXT:    [[TMP13:%.*]] = insertelement <4 x float> [[TMP12]], float [[TMP7]], i32 1
; CHECK-NEXT:    [[TMP14:%.*]] = insertelement <4 x float> [[TMP13]], float [[TMP9]], i32 2
; CHECK-NEXT:    [[TMP15:%.*]] = insertelement <4 x float> [[TMP14]], float [[TMP11]], i32 3
; CHECK:         store <4 x float> [[TMP15]], <4 x float>* {{.*}}, align 4
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
