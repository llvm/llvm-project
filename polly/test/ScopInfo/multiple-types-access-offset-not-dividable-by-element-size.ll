; RUN: opt %loadNPMPolly -polly-stmt-granularity=bb '-passes=print<polly-function-scops>' -pass-remarks-analysis="polly-scops" \
; RUN:     -polly-allow-differing-element-types \
; RUN:     -disable-output < %s  2>&1 | FileCheck %s
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
; CHECK:    Arrays {
; CHECK:        i8 MemRef_Short[*]; // Element size 1
; CHECK:        i8 MemRef_Float[*]; // Element size 1
; CHECK:        i8 MemRef_Double[*]; // Element size 1
; CHECK:    }
; CHECK:    Arrays (Bounds as pw_affs) {
; CHECK:        i8 MemRef_Short[*]; // Element size 1
; CHECK:        i8 MemRef_Float[*]; // Element size 1
; CHECK:        i8 MemRef_Double[*]; // Element size 1
; CHECK:    }
; CHECK:    Statements {
; CHECK:      Stmt_bb2
; CHECK:            Domain :=
; CHECK:                { Stmt_bb2[i0] : 0 <= i0 <= 99 };
; CHECK:            Schedule :=
; CHECK:                { Stmt_bb2[i0] -> [i0] };
; CHECK:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK:                { Stmt_bb2[i0] -> MemRef_Short[o0] : i0 <= o0 <= 1 + i0 };
; CHECK:            MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK:                { Stmt_bb2[i0] -> MemRef_Short[i0] };
; CHECK:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK:                { Stmt_bb2[i0] -> MemRef_Float[o0] : i0 <= o0 <= 3 + i0 };
; CHECK:            MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK:                { Stmt_bb2[i0] -> MemRef_Float[i0] };
; CHECK:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK:                { Stmt_bb2[i0] -> MemRef_Double[o0] : i0 <= o0 <= 7 + i0 };
; CHECK:            MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK:                { Stmt_bb2[i0] -> MemRef_Double[i0] };
; CHECK:    }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @multiple_types(ptr %Short, ptr %Float, ptr %Double) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb17, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp18, %bb17 ]
  %exitcond = icmp ne i64 %i.0, 100
  br i1 %exitcond, label %bb2, label %bb19

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds i8, ptr %Short, i64 %i.0
  %tmp4 = load i16, ptr %tmp, align 1
  %tmp5 = trunc i16 %tmp4 to i8
  %tmp6 = getelementptr inbounds i8, ptr %Short, i64 %i.0
  store i8 %tmp5, ptr %tmp6, align 1
  %tmp7 = getelementptr inbounds i8, ptr %Float, i64 %i.0
  %tmp9 = load float, ptr %tmp7, align 1
  %tmp10 = fptosi float %tmp9 to i8
  %tmp11 = getelementptr inbounds i8, ptr %Float, i64 %i.0
  store i8 %tmp10, ptr %tmp11, align 1
  %tmp12 = getelementptr inbounds i8, ptr %Double, i64 %i.0
  %tmp14 = load double, ptr %tmp12, align 1
  %tmp15 = fptosi double %tmp14 to i8
  %tmp16 = getelementptr inbounds i8, ptr %Double, i64 %i.0
  store i8 %tmp15, ptr %tmp16, align 1
  br label %bb17

bb17:                                             ; preds = %bb2
  %tmp18 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb19:                                             ; preds = %bb1
  ret void
}
