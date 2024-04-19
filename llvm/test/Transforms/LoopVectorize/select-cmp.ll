; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -S < %s | FileCheck %s --check-prefix=CHECK-VF4IC1 --check-prefix=CHECK
; RUN: opt -passes=loop-vectorize -force-vector-interleave=4 -force-vector-width=4 -S < %s | FileCheck %s --check-prefix=CHECK-VF4IC4 --check-prefix=CHECK
; RUN: opt -passes=loop-vectorize -force-vector-interleave=4 -force-vector-width=1 -S < %s | FileCheck %s --check-prefix=CHECK-VF1IC4 --check-prefix=CHECK

define i32 @select_const_i32_from_icmp(ptr nocapture readonly %v, i64 %n) {
; CHECK-LABEL: @select_const_i32_from_icmp
; CHECK-VF4IC1:      vector.body:
; CHECK-VF4IC1:        [[VEC_PHI:%.*]] = phi <4 x i1> [ zeroinitializer, %vector.ph ], [ [[VEC_SEL:%.*]], %vector.body ]
; CHECK-VF4IC1:        [[VEC_LOAD:%.*]] = load <4 x i32>
; CHECK-VF4IC1-NEXT:   [[VEC_ICMP:%.*]] = icmp eq <4 x i32> [[VEC_LOAD]], <i32 3, i32 3, i32 3, i32 3>
; CHECK-VF4IC1-NEXT:   [[NOT:%.*]] = xor <4 x i1> [[VEC_ICMP]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-VF4IC1-NEXT:   [[VEC_SEL]] = or <4 x i1> [[VEC_PHI]], [[NOT]]
; CHECK-VF4IC1:      middle.block:
; CHECK-VF4IC1-NEXT:   [[OR_RDX:%.*]] = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> [[VEC_SEL]])
; CHECK-VF4IC1-NEXT:   [[FR_OR_RDX:%.*]] = freeze i1 [[OR_RDX]]
; CHECK-VF4IC1-NEXT:   {{.*}} = select i1 [[FR_OR_RDX]], i32 7, i32 3

; CHECK-VF4IC4:      vector.body:
; CHECK-VF4IC4:        [[VEC_PHI1:%.*]] = phi <4 x i1> [ zeroinitializer, %vector.ph ], [ [[VEC_SEL1:%.*]], %vector.body ]
; CHECK-VF4IC4-NEXT:   [[VEC_PHI2:%.*]] = phi <4 x i1> [ zeroinitializer, %vector.ph ], [ [[VEC_SEL2:%.*]], %vector.body ]
; CHECK-VF4IC4-NEXT:   [[VEC_PHI3:%.*]] = phi <4 x i1> [ zeroinitializer, %vector.ph ], [ [[VEC_SEL3:%.*]], %vector.body ]
; CHECK-VF4IC4-NEXT:   [[VEC_PHI4:%.*]] = phi <4 x i1> [ zeroinitializer, %vector.ph ], [ [[VEC_SEL4:%.*]], %vector.body ]
; CHECK-VF4IC4:        [[VEC_ICMP1:%.*]] = icmp eq <4 x i32> {{.*}}, <i32 3, i32 3, i32 3, i32 3>
; CHECK-VF4IC4-NEXT:   [[VEC_ICMP2:%.*]] = icmp eq <4 x i32> {{.*}}, <i32 3, i32 3, i32 3, i32 3>
; CHECK-VF4IC4-NEXT:   [[VEC_ICMP3:%.*]] = icmp eq <4 x i32> {{.*}}, <i32 3, i32 3, i32 3, i32 3>
; CHECK-VF4IC4-NEXT:   [[VEC_ICMP4:%.*]] = icmp eq <4 x i32> {{.*}}, <i32 3, i32 3, i32 3, i32 3>
; CHECK-VF4IC4-NEXT:   [[NOT1:%.*]] = xor <4 x i1> [[VEC_ICMP1]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-VF4IC4-NEXT:   [[NOT2:%.*]] = xor <4 x i1> [[VEC_ICMP2]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-VF4IC4-NEXT:   [[NOT3:%.*]] = xor <4 x i1> [[VEC_ICMP3]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-VF4IC4-NEXT:   [[NOT4:%.*]] = xor <4 x i1> [[VEC_ICMP4]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-VF4IC4-NEXT:   [[VEC_SEL1:%.*]] = or <4 x i1> [[VEC_PHI1]], [[NOT1]]
; CHECK-VF4IC4-NEXT:   [[VEC_SEL2:%.*]] = or <4 x i1> [[VEC_PHI2]], [[NOT2]]
; CHECK-VF4IC4-NEXT:   [[VEC_SEL3:%.*]] = or <4 x i1> [[VEC_PHI3]], [[NOT3]]
; CHECK-VF4IC4-NEXT:   [[VEC_SEL4:%.*]] = or <4 x i1> [[VEC_PHI4]], [[NOT4]]
; CHECK-VF4IC4:      middle.block:
; CHECK-VF4IC4-NEXT:   [[VEC_SEL5:%.*]] = or <4 x i1>  [[VEC_SEL2]], [[VEC_SEL1]]
; CHECK-VF4IC4-NEXT:   [[VEC_SEL6:%.*]] = or <4 x i1> [[VEC_SEL3]], [[VEC_SEL5]]
; CHECK-VF4IC4-NEXT:   [[VEC_SEL7:%.*]] = or <4 x i1> [[VEC_SEL4]], [[VEC_SEL6]]
; CHECK-VF4IC4-NEXT:   [[OR_RDX:%.*]] = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> [[VEC_SEL7]])
; CHECK-VF4IC4-NEXT:   [[FR_OR_RDX:%.*]] = freeze i1 [[OR_RDX]]
; CHECK-VF4IC4-NEXT:   {{.*}} = select i1 [[FR_OR_RDX]], i32 7, i32 3


; CHECK-VF1IC4:      vector.body:
; CHECK-VF1IC4:        [[VEC_PHI1:%.*]] = phi i1 [ false, %vector.ph ], [ [[VEC_SEL1:%.*]], %vector.body ]
; CHECK-VF1IC4-NEXT:   [[VEC_PHI2:%.*]] = phi i1 [ false, %vector.ph ], [ [[VEC_SEL2:%.*]], %vector.body ]
; CHECK-VF1IC4-NEXT:   [[VEC_PHI3:%.*]] = phi i1 [ false, %vector.ph ], [ [[VEC_SEL3:%.*]], %vector.body ]
; CHECK-VF1IC4-NEXT:   [[VEC_PHI4:%.*]] = phi i1 [ false, %vector.ph ], [ [[VEC_SEL4:%.*]], %vector.body ]
; CHECK-VF1IC4:        [[VEC_LOAD1:%.*]] = load i32
; CHECK-VF1IC4-NEXT:   [[VEC_LOAD2:%.*]] = load i32
; CHECK-VF1IC4-NEXT:   [[VEC_LOAD3:%.*]] = load i32
; CHECK-VF1IC4-NEXT:   [[VEC_LOAD4:%.*]] = load i32
; CHECK-VF1IC4-NEXT:   [[VEC_ICMP1:%.*]] = icmp eq i32 [[VEC_LOAD1]], 3
; CHECK-VF1IC4-NEXT:   [[VEC_ICMP2:%.*]] = icmp eq i32 [[VEC_LOAD2]], 3
; CHECK-VF1IC4-NEXT:   [[VEC_ICMP3:%.*]] = icmp eq i32 [[VEC_LOAD3]], 3
; CHECK-VF1IC4-NEXT:   [[VEC_ICMP4:%.*]] = icmp eq i32 [[VEC_LOAD4]], 3
; CHECK-VF1IC4-NEXT:   [[NOT1:%.*]] = xor i1 [[VEC_ICMP1]], true
; CHECK-VF1IC4-NEXT:   [[NOT2:%.*]] = xor i1 [[VEC_ICMP2]], true
; CHECK-VF1IC4-NEXT:   [[NOT3:%.*]] = xor i1 [[VEC_ICMP3]], true
; CHECK-VF1IC4-NEXT:   [[NOT4:%.*]] = xor i1 [[VEC_ICMP4]], true
; CHECK-VF1IC4-NEXT:   [[VEC_SEL1:%.*]] = or i1 [[VEC_PHI1]], [[NOT1]]
; CHECK-VF1IC4-NEXT:   [[VEC_SEL2:%.*]] = or i1 [[VEC_PHI2]], [[NOT2]]
; CHECK-VF1IC4-NEXT:   [[VEC_SEL3:%.*]] = or i1 [[VEC_PHI3]], [[NOT3]]
; CHECK-VF1IC4-NEXT:   [[VEC_SEL4:%.*]] = or i1 [[VEC_PHI4]], [[NOT4]]
; CHECK-VF1IC4:      middle.block:
; CHECK-VF1IC4-NEXT:   [[VEC_SEL5:%.*]] = or i1 [[VEC_SEL2]], [[VEC_SEL1]]
; CHECK-VF1IC4-NEXT:   [[VEC_SEL6:%.*]] = or i1 [[VEC_SEL3]], [[VEC_SEL5]]
; CHECK-VF1IC4-NEXT:   [[OR_RDX:%.*]] = or i1  [[VEC_SEL4]], [[VEC_SEL6]]
; CHECK-VF1IC4-NEXT:   [[FR_OR_RDX:%.*]] = freeze i1 [[OR_RDX]]
; CHECK-VF1IC4-NEXT:   {{.*}} = select i1 [[FR_OR_RDX]], i32 7, i32 3

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
  br i1 %7, label %exit, label %for.body

exit:                                     ; preds = %for.body
  ret i32 %5
}


define i32 @select_const_i32_from_icmp2(ptr nocapture readonly %v, i64 %n) {
; CHECK-LABEL: @select_const_i32_from_icmp2
; CHECK-VF4IC1:      vector.body:
; CHECK-VF4IC1:        [[VEC_PHI:%.*]] = phi <4 x i1> [ zeroinitializer, %vector.ph ], [ [[VEC_SEL:%.*]], %vector.body ]
; CHECK-VF4IC1:        [[VEC_LOAD:%.*]] = load <4 x i32>
; CHECK-VF4IC1-NEXT:   [[VEC_ICMP:%.*]] = icmp eq <4 x i32> [[VEC_LOAD]], <i32 3, i32 3, i32 3, i32 3>
; CHECK-VF4IC1-NEXT:   [[VEC_SEL]] = or <4 x i1> [[VEC_PHI]], [[VEC_ICMP]]
; CHECK-VF4IC1:      middle.block:
; CHECK-VF4IC1-NEXT:   [[OR_RDX:%.*]] = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> [[VEC_SEL]])
; CHECK-VF4IC1-NEXT:   [[FR_OR_RDX:%.*]] = freeze i1 [[OR_RDX]]
; CHECK-VF4IC1-NEXT:   {{.*}} = select i1 [[FR_OR_RDX]], i32 7, i32 3

entry:
  br label %for.body

for.body:                                      ; preds = %entry, %for.body
  %0 = phi i64 [ 0, %entry ], [ %6, %for.body ]
  %1 = phi i32 [ 3, %entry ], [ %5, %for.body ]
  %2 = getelementptr inbounds i32, ptr %v, i64 %0
  %3 = load i32, ptr %2, align 4
  %4 = icmp eq i32 %3, 3
  %5 = select i1 %4, i32 7, i32 %1
  %6 = add nuw nsw i64 %0, 1
  %7 = icmp eq i64 %6, %n
  br i1 %7, label %exit, label %for.body

exit:                                     ; preds = %for.body
  ret i32 %5
}


define i32 @select_i32_from_icmp(ptr nocapture readonly %v, i32 %a, i32 %b, i64 %n) {
; CHECK-LABEL: @select_i32_from_icmp
; CHECK-VF4IC1:      vector.ph:
; CHECK-VF4IC1-NOT:    shufflevector <4 x i32>
; CHECK-VF4IC1-NOT:    shufflevector <4 x i32>
; CHECK-VF4IC1:      vector.body:
; CHECK-VF4IC1:        [[VEC_PHI:%.*]] = phi <4 x i1> [ zeroinitializer, %vector.ph ], [ [[VEC_SEL:%.*]], %vector.body ]
; CHECK-VF4IC1:        [[VEC_LOAD:%.*]] = load <4 x i32>
; CHECK-VF4IC1-NEXT:   [[VEC_ICMP:%.*]] = icmp eq <4 x i32> [[VEC_LOAD]], <i32 3, i32 3, i32 3, i32 3>
; CHECK-VF4IC1-NEXT:   [[NOT:%.*]] = xor <4 x i1> [[VEC_ICMP]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-VF4IC1-NEXT:   [[VEC_SEL]] = or <4 x i1> [[VEC_PHI]], [[NOT]]
; CHECK-VF4IC1:      middle.block:
; CHECK-VF4IC1-NEXT:   [[OR_RDX:%.*]] = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> [[VEC_SEL]])
; CHECK-VF4IC1-NEXT:   [[FR_OR_RDX:%.*]] = freeze i1 [[OR_RDX]]
; CHECK-VF4IC1-NEXT:   {{.*}} = select i1 [[FR_OR_RDX]], i32 %b, i32 %a
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
  br i1 %7, label %exit, label %for.body

exit:                                     ; preds = %for.body
  ret i32 %5
}


define i32 @select_const_i32_from_fcmp_fast(ptr nocapture readonly %v, i64 %n) {
; CHECK-LABEL: @select_const_i32_from_fcmp_fast
; CHECK-VF4IC1:      vector.body:
; CHECK-VF4IC1:        [[VEC_PHI:%.*]] = phi <4 x i1> [ zeroinitializer, %vector.ph ], [ [[VEC_SEL:%.*]], %vector.body ]
; CHECK-VF4IC1:        [[VEC_LOAD:%.*]] = load <4 x float>
; CHECK-VF4IC1-NEXT:   [[VEC_FCMP:%.*]] = fcmp fast ueq <4 x float> [[VEC_LOAD]], <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
; CHECK-VF4IC1-NEXT:   [[NOT:%.*]] = xor <4 x i1> [[VEC_FCMP]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-VF4IC1-NEXT:   [[VEC_SEL]] = or <4 x i1> [[VEC_PHI]], [[NOT]]
; CHECK-VF4IC1:      middle.block:
; CHECK-VF4IC1-NEXT:   [[OR_RDX:%.*]] = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> [[VEC_SEL]])
; CHECK-VF4IC1-NEXT:   [[FR_OR_RDX:%.*]] = freeze i1 [[OR_RDX]]
; CHECK-VF4IC1-NEXT:   {{.*}} = select i1 [[FR_OR_RDX]], i32 1, i32 2
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
  br i1 %7, label %exit, label %for.body

exit:                                     ; preds = %for.body
  ret i32 %5
}


define i32 @select_const_i32_from_fcmp(ptr nocapture readonly %v, i64 %n) {
; CHECK-LABEL: @select_const_i32_from_fcmp
; CHECK-VF4IC1:      vector.body:
; CHECK-VF4IC1:        [[VEC_PHI:%.*]] = phi <4 x i1> [ zeroinitializer, %vector.ph ], [ [[VEC_SEL:%.*]], %vector.body ]
; CHECK-VF4IC1:        [[VEC_LOAD:%.*]] = load <4 x float>
; CHECK-VF4IC1-NEXT:   [[VEC_FCMP:%.*]] = fcmp ueq <4 x float> [[VEC_LOAD]], <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
; CHECK-VF4IC1-NEXT:   [[NOT:%.*]] = xor <4 x i1> [[VEC_FCMP]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-VF4IC1-NEXT:   [[VEC_SEL]] = or <4 x i1> [[VEC_PHI]], [[NOT]]
; CHECK-VF4IC1:      middle.block:
; CHECK-VF4IC1-NEXT:   [[OR_RDX:%.*]] = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> [[VEC_SEL]])
; CHECK-VF4IC1-NEXT:   [[FR_OR_RDX:%.*]] = freeze i1 [[OR_RDX]]
; CHECK-VF4IC1-NEXT:   {{.*}} = select i1 [[FR_OR_RDX]], i32 1, i32 2
entry:
  br label %for.body

for.body:                                      ; preds = %entry, %for.body
  %0 = phi i64 [ 0, %entry ], [ %6, %for.body ]
  %1 = phi i32 [ 2, %entry ], [ %5, %for.body ]
  %2 = getelementptr inbounds float, ptr %v, i64 %0
  %3 = load float, ptr %2, align 4
  %4 = fcmp ueq float %3, 3.0
  %5 = select i1 %4, i32 %1, i32 1
  %6 = add nuw nsw i64 %0, 1
  %7 = icmp eq i64 %6, %n
  br i1 %7, label %exit, label %for.body

exit:                                     ; preds = %for.body
  ret i32 %5
}


define i32 @select_i32_from_icmp_same_inputs(i32 %a, i32 %b, i64 %n) {
; CHECK-LABEL: @select_i32_from_icmp_same_inputs
; CHECK-VF4IC1:      vector.ph:
; CHECK-VF4IC1:        [[TMP1:%.*]] = insertelement <4 x i32> poison, i32 %a, i64 0
; CHECK-VF4IC1-NEXT:   [[SPLAT_OF_A:%.*]] = shufflevector <4 x i32> [[TMP1]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK-VF4IC1-NOT:   [[TMP2:%.*]] = insertelement <4 x i32> poison, i32 %b, i64 0
; CHECK-VF4IC1:      vector.body:
; CHECK-VF4IC1:        [[VEC_PHI:%.*]] = phi <4 x i1> [ zeroinitializer, %vector.ph ], [ [[VEC_SEL:%.*]], %vector.body ]
; CHECK-VF4IC1:        [[VEC_ICMP:%.*]] = icmp eq <4 x i32> [[SPLAT_OF_A]], <i32 3, i32 3, i32 3, i32 3>
; CHECK-VF4IC1-NEXT:   [[NOT:%.*]] = xor <4 x i1> [[VEC_ICMP]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-VF4IC1-NEXT:   [[VEC_SEL]] = or <4 x i1> [[VEC_PHI]], [[NOT]]
; CHECK-VF4IC1:      middle.block:
; CHECK-VF4IC1-NEXT:   [[OR_RDX:%.*]] = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> [[VEC_SEL]])
; CHECK-VF4IC1-NEXT:   [[FR_OR_RDX:%.*]] = freeze i1 [[OR_RDX]]
; CHECK-VF4IC1-NEXT:   {{.*}} = select i1 [[FR_OR_RDX]], i32 %b, i32 %a
entry:
  br label %for.body

for.body:                                      ; preds = %entry, %for.body
  %0 = phi i64 [ 0, %entry ], [ %4, %for.body ]
  %1 = phi i32 [ %a, %entry ], [ %3, %for.body ]
  %2 = icmp eq i32 %1, 3
  %3 = select i1 %2, i32 %1, i32 %b
  %4 = add nuw nsw i64 %0, 1
  %5 = icmp eq i64 %4, %n
  br i1 %5, label %exit, label %for.body

exit:                                     ; preds = %for.body
  ret i32 %3
}


; Negative tests

; We don't support FP reduction variables at the moment.
define float @select_const_f32_from_icmp(ptr nocapture readonly %v, i64 %n) {
; CHECK: @select_const_f32_from_icmp
; CHECK-NOT: vector.body
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
  br i1 %7, label %exit, label %for.body

exit:                                     ; preds = %for.body
  ret float %5
}


; We don't support select/cmp reduction patterns where there is more than one
; use of the icmp/fcmp.
define i32 @select_const_i32_from_icmp_mul_use(ptr nocapture readonly %v1, ptr %v2, i64 %n) {
; CHECK-LABEL: @select_const_i32_from_icmp_mul_use
; CHECK-NOT: vector.body
entry:
  br label %for.body

for.body:                                      ; preds = %entry, %for.body
  %0 = phi i64 [ 0, %entry ], [ %8, %for.body ]
  %1 = phi i32 [ 3, %entry ], [ %6, %for.body ]
  %2 = phi i32 [ 0, %entry ], [ %7, %for.body ]
  %3 = getelementptr inbounds i32, ptr %v1, i64 %0
  %4 = load i32, ptr %3, align 4
  %5 = icmp eq i32 %4, 3
  %6 = select i1 %5, i32 %1, i32 7
  %7 = zext i1 %5 to i32
  %8 = add nuw nsw i64 %0, 1
  %9 = icmp eq i64 %8, %n
  br i1 %9, label %exit, label %for.body

exit:                                     ; preds = %for.body
  store i32 %7, ptr %v2, align 4
  ret i32 %6
}


; We don't support selecting loop-variant values.
define i32 @select_variant_i32_from_icmp(ptr nocapture readonly %v1, ptr nocapture readonly %v2, i64 %n) {
; CHECK-LABEL: @select_variant_i32_from_icmp
; CHECK-NOT: vector.body
entry:
  br label %for.body

for.body:                                      ; preds = %entry, %for.body
  %0 = phi i64 [ 0, %entry ], [ %8, %for.body ]
  %1 = phi i32 [ 3, %entry ], [ %7, %for.body ]
  %2 = getelementptr inbounds i32, ptr %v1, i64 %0
  %3 = load i32, ptr %2, align 4
  %4 = getelementptr inbounds i32, ptr %v2, i64 %0
  %5 = load i32, ptr %4, align 4
  %6 = icmp eq i32 %3, 3
  %7 = select i1 %6, i32 %1, i32 %5
  %8 = add nuw nsw i64 %0, 1
  %9 = icmp eq i64 %8, %n
  br i1 %9, label %exit, label %for.body

exit:                                     ; preds = %for.body
  ret i32 %7
}


; We only support selects where the input comes from the same PHI as the
; reduction PHI. In the example below, the select uses the induction
; variable input and the icmp uses the reduction PHI.
define i32 @select_i32_from_icmp_non_redux_phi(i32 %a, i32 %b, i32 %n) {
; CHECK-LABEL: @select_i32_from_icmp_non_redux_phi
; CHECK-NOT: vector.body
entry:
  br label %for.body

for.body:                                      ; preds = %entry, %for.body
  %0 = phi i32 [ 0, %entry ], [ %4, %for.body ]
  %1 = phi i32 [ %a, %entry ], [ %3, %for.body ]
  %2 = icmp eq i32 %1, 3
  %3 = select i1 %2, i32 %0, i32 %b
  %4 = add nuw nsw i32 %0, 1
  %5 = icmp eq i32 %4, %n
  br i1 %5, label %exit, label %for.body

exit:                                     ; preds = %for.body
  ret i32 %3
}
