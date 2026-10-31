; RUN: opt < %s -passes=slp-vectorizer -S -slp-min-reg-size=64 -slp-max-reg-size=64 -slp-threshold=-99999 | FileCheck %s

define void @sext_add_nsw(ptr %p, i1 %c, i32 %i0) {
; CHECK-LABEL: @sext_add_nsw(
; CHECK:       loop:
; CHECK-NEXT:    [[IV:%.*]] = phi i32 [ [[I0:%.*]], %entry ], [ [[NEXT:%.*]], %loop ]
; CHECK-NEXT:    [[OFFSET0:%.*]] = mul i32 [[IV]], 2
; CHECK-NEXT:    [[INDEX0:%.*]] = sext i32 [[OFFSET0]] to i64
; CHECK-NEXT:    [[PTR0:%.*]] = getelementptr inbounds float, ptr [[P:%.*]], i64 [[INDEX0]]
; CHECK-NEXT:    [[VLOAD:%.*]] = load <2 x float>, ptr [[PTR0]], align 4
; CHECK-NEXT:    [[VADD:%.*]] = fadd <2 x float> [[VLOAD]], splat (float 1.000000e+00)
; CHECK-NEXT:    store <2 x float> [[VADD]], ptr [[PTR0]], align 4
; CHECK-NEXT:    [[NEXT]] = add i32 [[IV]], 1
; CHECK-NEXT:    br i1 [[C:%.*]], label %loop, label %exit
entry:
  br label %loop

loop:
  %iv = phi i32 [ %i0, %entry ], [ %next, %loop ]
  %offset0 = mul i32 %iv, 2
  %index0 = sext i32 %offset0 to i64
  %ptr0 = getelementptr inbounds float, ptr %p, i64 %index0
  %value0 = load float, ptr %ptr0, align 4
  %increment0 = fadd float %value0, 1.000000e+00
  store float %increment0, ptr %ptr0, align 4
  %offset1 = add nsw i32 %offset0, 1
  %index1 = sext i32 %offset1 to i64
  %ptr1 = getelementptr inbounds float, ptr %p, i64 %index1
  %value1 = load float, ptr %ptr1, align 4
  %increment1 = fadd float %value1, 1.000000e+00
  store float %increment1, ptr %ptr1, align 4
  %next = add i32 %iv, 1
  br i1 %c, label %loop, label %exit

exit:
  ret void
}

define void @zext_add_nuw(ptr %p, i1 %c, i32 %i0) {
; CHECK-LABEL: @zext_add_nuw(
; CHECK:       loop:
; CHECK-NEXT:    [[IV:%.*]] = phi i32 [ [[I0:%.*]], %entry ], [ [[NEXT:%.*]], %loop ]
; CHECK-NEXT:    [[OFFSET0:%.*]] = mul i32 [[IV]], 2
; CHECK-NEXT:    [[INDEX0:%.*]] = zext i32 [[OFFSET0]] to i64
; CHECK-NEXT:    [[PTR0:%.*]] = getelementptr inbounds float, ptr [[P:%.*]], i64 [[INDEX0]]
; CHECK-NEXT:    [[VLOAD:%.*]] = load <2 x float>, ptr [[PTR0]], align 4
; CHECK-NEXT:    [[VADD:%.*]] = fadd <2 x float> [[VLOAD]], splat (float 1.000000e+00)
; CHECK-NEXT:    store <2 x float> [[VADD]], ptr [[PTR0]], align 4
; CHECK-NEXT:    [[NEXT]] = add i32 [[IV]], 1
; CHECK-NEXT:    br i1 [[C:%.*]], label %loop, label %exit
entry:
  br label %loop

loop:
  %iv = phi i32 [ %i0, %entry ], [ %next, %loop ]
  %offset0 = mul i32 %iv, 2
  %index0 = zext i32 %offset0 to i64
  %ptr0 = getelementptr inbounds float, ptr %p, i64 %index0
  %value0 = load float, ptr %ptr0, align 4
  %increment0 = fadd float %value0, 1.000000e+00
  store float %increment0, ptr %ptr0, align 4
  %offset1 = add nuw i32 %offset0, 1
  %index1 = zext i32 %offset1 to i64
  %ptr1 = getelementptr inbounds float, ptr %p, i64 %index1
  %value1 = load float, ptr %ptr1, align 4
  %increment1 = fadd float %value1, 1.000000e+00
  store float %increment1, ptr %ptr1, align 4
  %next = add i32 %iv, 1
  br i1 %c, label %loop, label %exit

exit:
  ret void
}
