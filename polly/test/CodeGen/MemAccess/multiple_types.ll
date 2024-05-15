; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-import-jscop \
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
; CHECK: %polly.access.Short10 = getelementptr i8, ptr %Short, i64 0
; CHECK: %tmp5_p_scalar_ = load i16, ptr %polly.access.Short10

; Float[8 * i]
; CHECK: %24 = mul nsw i64 8, %polly.indvar
; CHECK: %polly.access.Float11 = getelementptr i8, ptr %Float, i64 %24
; CHECK: %tmp11_p_scalar_ = load float, ptr %polly.access.Float11

; Double[8]
; CHECK: %polly.access.Double13 = getelementptr i8, ptr %Double, i64 8
; CHECK: %tmp17_p_scalar_ = load double, ptr %polly.access.Double13

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @multiple_types(ptr %Short, ptr %Float, ptr %Double) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb20, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp21, %bb20 ]
  %exitcond = icmp ne i64 %i.0, 100
  br i1 %exitcond, label %bb2, label %bb22

bb2:                                              ; preds = %bb1
  %tmp = shl nsw i64 %i.0, 1
  %tmp3 = getelementptr inbounds i8, ptr %Short, i64 %tmp
  %tmp5 = load i16, ptr %tmp3, align 2
  %tmp6 = trunc i16 %tmp5 to i8
  %tmp7 = getelementptr inbounds i8, ptr %Short, i64 %i.0
  store i8 %tmp6, ptr %tmp7, align 1
  %tmp8 = shl nsw i64 %i.0, 2
  %tmp9 = getelementptr inbounds i8, ptr %Float, i64 %tmp8
  %tmp11 = load float, ptr %tmp9, align 4
  %tmp12 = fptosi float %tmp11 to i8
  %tmp13 = getelementptr inbounds i8, ptr %Float, i64 %i.0
  store i8 %tmp12, ptr %tmp13, align 1
  %tmp14 = shl nsw i64 %i.0, 3
  %tmp15 = getelementptr inbounds i8, ptr %Double, i64 %tmp14
  %tmp17 = load double, ptr %tmp15, align 8
  %tmp18 = fptosi double %tmp17 to i8
  %tmp19 = getelementptr inbounds i8, ptr %Double, i64 %i.0
  store i8 %tmp18, ptr %tmp19, align 1
  br label %bb20

bb20:                                             ; preds = %bb2
  %tmp21 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb22:                                             ; preds = %bb1
  ret void
}
