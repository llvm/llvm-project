; RUN: opt %loadPolly -polly-scops -pass-remarks-analysis="polly-scops" \
; RUN:                -analyze < %s  2>&1 | FileCheck %s
;
;    // For the following accesses the offset expression from the base pointer
;    // is not always a multiple of the type size.
;    void multiple_types(char *Short, char *Float, char *Double) {
;      for (long i = 0; i < 100; i++) {
;        Short[i] = *(short *)&Short[i];
;        Float[i] = *(float *)&Float[i];
;        Double[i] = *(double *)&Double[i];
;      }
;    }
;
;   Polly currently does not allow such cases (even without multiple accesses of
;   different type being involved).
;   TODO: Add support for such kind of accesses
;
;
; CHECK: Alignment assumption: {  : 1 = 0 }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @multiple_types(i8* %Short, i8* %Float, i8* %Double) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb17, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp18, %bb17 ]
  %exitcond = icmp ne i64 %i.0, 100
  br i1 %exitcond, label %bb2, label %bb19

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds i8, i8* %Short, i64 %i.0
  %tmp3 = bitcast i8* %tmp to i16*
  %tmp4 = load i16, i16* %tmp3, align 1
  %tmp5 = trunc i16 %tmp4 to i8
  %tmp6 = getelementptr inbounds i8, i8* %Short, i64 %i.0
  store i8 %tmp5, i8* %tmp6, align 1
  %tmp7 = getelementptr inbounds i8, i8* %Float, i64 %i.0
  %tmp8 = bitcast i8* %tmp7 to float*
  %tmp9 = load float, float* %tmp8, align 1
  %tmp10 = fptosi float %tmp9 to i8
  %tmp11 = getelementptr inbounds i8, i8* %Float, i64 %i.0
  store i8 %tmp10, i8* %tmp11, align 1
  %tmp12 = getelementptr inbounds i8, i8* %Double, i64 %i.0
  %tmp13 = bitcast i8* %tmp12 to double*
  %tmp14 = load double, double* %tmp13, align 1
  %tmp15 = fptosi double %tmp14 to i8
  %tmp16 = getelementptr inbounds i8, i8* %Double, i64 %i.0
  store i8 %tmp15, i8* %tmp16, align 1
  br label %bb17

bb17:                                             ; preds = %bb2
  %tmp18 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb19:                                             ; preds = %bb1
  ret void
}
