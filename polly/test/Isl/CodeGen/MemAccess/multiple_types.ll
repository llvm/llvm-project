; RUN: opt %loadPolly -polly-import-jscop \
; RUN: -polly-allow-differing-element-types \
; RUN:   -polly-codegen -S    < %s | FileCheck %s
;
;    // Check that accessing one array with different types works.
;    void multiple_types(char *Short, char *Float, char *Double) {
;      for (long i = 0; i < 100; i++) {
;        Short[i] = *(short *)&Short[2 * i];
;        Float[i] = *(float *)&Float[4 * i];
;        Double[i] = *(double *)&Double[8 * i];
;      }
;    }

; Short[0]
; CHECK: %polly.access.Short10 = getelementptr i8, i8* %Short, i64 0
; CHECK: %24 = bitcast i8* %polly.access.Short10 to i16*
; CHECK: %tmp5_p_scalar_ = load i16, i16* %24

; Float[8 * i]
; CHECK: %25 = mul nsw i64 8, %polly.indvar
; CHECK: %polly.access.Float11 = getelementptr i8, i8* %Float, i64 %25
; CHECK: %26 = bitcast i8* %polly.access.Float11 to float*
; CHECK: %tmp11_p_scalar_ = load float, float* %26

; Double[8]
; CHECK: %polly.access.Double13 = getelementptr i8, i8* %Double, i64 8
; CHECK: %27 = bitcast i8* %polly.access.Double13 to double*
; CHECK: %tmp17_p_scalar_ = load double, double* %27

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @multiple_types(i8* %Short, i8* %Float, i8* %Double) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb20, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp21, %bb20 ]
  %exitcond = icmp ne i64 %i.0, 100
  br i1 %exitcond, label %bb2, label %bb22

bb2:                                              ; preds = %bb1
  %tmp = shl nsw i64 %i.0, 1
  %tmp3 = getelementptr inbounds i8, i8* %Short, i64 %tmp
  %tmp4 = bitcast i8* %tmp3 to i16*
  %tmp5 = load i16, i16* %tmp4, align 2
  %tmp6 = trunc i16 %tmp5 to i8
  %tmp7 = getelementptr inbounds i8, i8* %Short, i64 %i.0
  store i8 %tmp6, i8* %tmp7, align 1
  %tmp8 = shl nsw i64 %i.0, 2
  %tmp9 = getelementptr inbounds i8, i8* %Float, i64 %tmp8
  %tmp10 = bitcast i8* %tmp9 to float*
  %tmp11 = load float, float* %tmp10, align 4
  %tmp12 = fptosi float %tmp11 to i8
  %tmp13 = getelementptr inbounds i8, i8* %Float, i64 %i.0
  store i8 %tmp12, i8* %tmp13, align 1
  %tmp14 = shl nsw i64 %i.0, 3
  %tmp15 = getelementptr inbounds i8, i8* %Double, i64 %tmp14
  %tmp16 = bitcast i8* %tmp15 to double*
  %tmp17 = load double, double* %tmp16, align 8
  %tmp18 = fptosi double %tmp17 to i8
  %tmp19 = getelementptr inbounds i8, i8* %Double, i64 %i.0
  store i8 %tmp18, i8* %tmp19, align 1
  br label %bb20

bb20:                                             ; preds = %bb2
  %tmp21 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb22:                                             ; preds = %bb1
  ret void
}
