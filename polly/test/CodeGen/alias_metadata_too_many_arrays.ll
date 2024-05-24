; RUN: opt %loadNPMPolly -passes=polly-codegen -polly-ignore-aliasing -S < %s \
; RUN:   | FileCheck %s
;
;    void manyarrays(float A1[], float A2[], float A3[], float A4[], float A5[],
;                    float A6[], float A7[], float A8[], float A9[]) {
;      for (long i = 0; i <= 1024; i++) {
;        A1[i] += i;
;        A2[i] += i;
;        A3[i] += i;
;        A4[i] += i;
;        A5[i] += i;
;        A6[i] += i;
;        A7[i] += i;
;        A8[i] += i;
;        A9[i] += i;
;      }
;    }
;
; CHECK-LABEL @manyarrays
; CHECK: load{{.*}}!alias.scope
; CHECK: store{{.*}}!alias.scope
; CHECK: load{{.*}}!alias.scope
; CHECK: store{{.*}}!alias.scope
; CHECK: load{{.*}}!alias.scope
; CHECK: store{{.*}}!alias.scope
; CHECK: load{{.*}}!alias.scope
; CHECK: store{{.*}}!alias.scope
; CHECK: load{{.*}}!alias.scope
; CHECK: store{{.*}}!alias.scope
; CHECK: load{{.*}}!alias.scope
; CHECK: store{{.*}}!alias.scope
; CHECK: load{{.*}}!alias.scope
; CHECK: store{{.*}}!alias.scope
; CHECK: load{{.*}}!alias.scope
; CHECK: store{{.*}}!alias.scope
; CHECK: load{{.*}}!alias.scope
; CHECK: store{{.*}}!alias.scope
;
;    void toomanyarrays(float A1[], float A2[], float A3[], float A4[], float A5[],
;                       float A6[], float A7[], float A8[], float A9[], float A10[],
;                       float A11[]) {
;      for (long i = 0; i <= 1024; i++) {
;        A1[i] += i;
;        A2[i] += i;
;        A3[i] += i;
;        A4[i] += i;
;        A5[i] += i;
;        A6[i] += i;
;        A7[i] += i;
;        A8[i] += i;
;        A9[i] += i;
;        A10[i] += i;
;        A11[i] += i;
;      }
;    }
;
; CHECK-LABEL: @toomanyarrays
; CHECK-NOT: !alias.scope
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @manyarrays(ptr %A1, ptr %A2, ptr %A3, ptr %A4, ptr %A5, ptr %A6, ptr %A7, ptr %A8, ptr %A9) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb38, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp39, %bb38 ]
  %exitcond = icmp ne i64 %i.0, 1025
  br i1 %exitcond, label %bb2, label %bb40

bb2:                                              ; preds = %bb1
  %tmp = sitofp i64 %i.0 to float
  %tmp3 = getelementptr inbounds float, ptr %A1, i64 %i.0
  %tmp4 = load float, ptr %tmp3, align 4
  %tmp5 = fadd float %tmp4, %tmp
  store float %tmp5, ptr %tmp3, align 4
  %tmp6 = sitofp i64 %i.0 to float
  %tmp7 = getelementptr inbounds float, ptr %A2, i64 %i.0
  %tmp8 = load float, ptr %tmp7, align 4
  %tmp9 = fadd float %tmp8, %tmp6
  store float %tmp9, ptr %tmp7, align 4
  %tmp10 = sitofp i64 %i.0 to float
  %tmp11 = getelementptr inbounds float, ptr %A3, i64 %i.0
  %tmp12 = load float, ptr %tmp11, align 4
  %tmp13 = fadd float %tmp12, %tmp10
  store float %tmp13, ptr %tmp11, align 4
  %tmp14 = sitofp i64 %i.0 to float
  %tmp15 = getelementptr inbounds float, ptr %A4, i64 %i.0
  %tmp16 = load float, ptr %tmp15, align 4
  %tmp17 = fadd float %tmp16, %tmp14
  store float %tmp17, ptr %tmp15, align 4
  %tmp18 = sitofp i64 %i.0 to float
  %tmp19 = getelementptr inbounds float, ptr %A5, i64 %i.0
  %tmp20 = load float, ptr %tmp19, align 4
  %tmp21 = fadd float %tmp20, %tmp18
  store float %tmp21, ptr %tmp19, align 4
  %tmp22 = sitofp i64 %i.0 to float
  %tmp23 = getelementptr inbounds float, ptr %A6, i64 %i.0
  %tmp24 = load float, ptr %tmp23, align 4
  %tmp25 = fadd float %tmp24, %tmp22
  store float %tmp25, ptr %tmp23, align 4
  %tmp26 = sitofp i64 %i.0 to float
  %tmp27 = getelementptr inbounds float, ptr %A7, i64 %i.0
  %tmp28 = load float, ptr %tmp27, align 4
  %tmp29 = fadd float %tmp28, %tmp26
  store float %tmp29, ptr %tmp27, align 4
  %tmp30 = sitofp i64 %i.0 to float
  %tmp31 = getelementptr inbounds float, ptr %A8, i64 %i.0
  %tmp32 = load float, ptr %tmp31, align 4
  %tmp33 = fadd float %tmp32, %tmp30
  store float %tmp33, ptr %tmp31, align 4
  %tmp34 = sitofp i64 %i.0 to float
  %tmp35 = getelementptr inbounds float, ptr %A9, i64 %i.0
  %tmp36 = load float, ptr %tmp35, align 4
  %tmp37 = fadd float %tmp36, %tmp34
  store float %tmp37, ptr %tmp35, align 4
  br label %bb38

bb38:                                             ; preds = %bb2
  %tmp39 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb40:                                             ; preds = %bb1
  ret void
}

define void @toomanyarrays(ptr %A1, ptr %A2, ptr %A3, ptr %A4, ptr %A5, ptr %A6, ptr %A7, ptr %A8, ptr %A9, ptr %A10, ptr %A11) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb46, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp47, %bb46 ]
  %exitcond = icmp ne i64 %i.0, 1025
  br i1 %exitcond, label %bb2, label %bb48

bb2:                                              ; preds = %bb1
  %tmp = sitofp i64 %i.0 to float
  %tmp3 = getelementptr inbounds float, ptr %A1, i64 %i.0
  %tmp4 = load float, ptr %tmp3, align 4
  %tmp5 = fadd float %tmp4, %tmp
  store float %tmp5, ptr %tmp3, align 4
  %tmp6 = sitofp i64 %i.0 to float
  %tmp7 = getelementptr inbounds float, ptr %A2, i64 %i.0
  %tmp8 = load float, ptr %tmp7, align 4
  %tmp9 = fadd float %tmp8, %tmp6
  store float %tmp9, ptr %tmp7, align 4
  %tmp10 = sitofp i64 %i.0 to float
  %tmp11 = getelementptr inbounds float, ptr %A3, i64 %i.0
  %tmp12 = load float, ptr %tmp11, align 4
  %tmp13 = fadd float %tmp12, %tmp10
  store float %tmp13, ptr %tmp11, align 4
  %tmp14 = sitofp i64 %i.0 to float
  %tmp15 = getelementptr inbounds float, ptr %A4, i64 %i.0
  %tmp16 = load float, ptr %tmp15, align 4
  %tmp17 = fadd float %tmp16, %tmp14
  store float %tmp17, ptr %tmp15, align 4
  %tmp18 = sitofp i64 %i.0 to float
  %tmp19 = getelementptr inbounds float, ptr %A5, i64 %i.0
  %tmp20 = load float, ptr %tmp19, align 4
  %tmp21 = fadd float %tmp20, %tmp18
  store float %tmp21, ptr %tmp19, align 4
  %tmp22 = sitofp i64 %i.0 to float
  %tmp23 = getelementptr inbounds float, ptr %A6, i64 %i.0
  %tmp24 = load float, ptr %tmp23, align 4
  %tmp25 = fadd float %tmp24, %tmp22
  store float %tmp25, ptr %tmp23, align 4
  %tmp26 = sitofp i64 %i.0 to float
  %tmp27 = getelementptr inbounds float, ptr %A7, i64 %i.0
  %tmp28 = load float, ptr %tmp27, align 4
  %tmp29 = fadd float %tmp28, %tmp26
  store float %tmp29, ptr %tmp27, align 4
  %tmp30 = sitofp i64 %i.0 to float
  %tmp31 = getelementptr inbounds float, ptr %A8, i64 %i.0
  %tmp32 = load float, ptr %tmp31, align 4
  %tmp33 = fadd float %tmp32, %tmp30
  store float %tmp33, ptr %tmp31, align 4
  %tmp34 = sitofp i64 %i.0 to float
  %tmp35 = getelementptr inbounds float, ptr %A9, i64 %i.0
  %tmp36 = load float, ptr %tmp35, align 4
  %tmp37 = fadd float %tmp36, %tmp34
  store float %tmp37, ptr %tmp35, align 4
  %tmp38 = sitofp i64 %i.0 to float
  %tmp39 = getelementptr inbounds float, ptr %A10, i64 %i.0
  %tmp40 = load float, ptr %tmp39, align 4
  %tmp41 = fadd float %tmp40, %tmp38
  store float %tmp41, ptr %tmp39, align 4
  %tmp42 = sitofp i64 %i.0 to float
  %tmp43 = getelementptr inbounds float, ptr %A11, i64 %i.0
  %tmp44 = load float, ptr %tmp43, align 4
  %tmp45 = fadd float %tmp44, %tmp42
  store float %tmp45, ptr %tmp43, align 4
  br label %bb46

bb46:                                             ; preds = %bb2
  %tmp47 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb48:                                             ; preds = %bb1
  ret void
}
