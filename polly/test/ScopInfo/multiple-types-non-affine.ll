; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-allow-differing-element-types -polly-print-scops -polly-allow-nonaffine -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-allow-differing-element-types -polly-codegen -polly-allow-nonaffine -disable-output
;
;    // Check that accessing one array with different types works,
;    // even though some accesses are non-affine.
;    void multiple_types(char *Short, short *Float, char *Double) {
;      for (long i = 0; i < 100; i++) {
;        Short[i] = *(short *)&Short[i & 8];
;        Float[i] = *(float *)&Float[i & 8];
;        Double[i] = *(double *)&Double[i & 8];
;      }
;    }
;
; CHECK:    Arrays {
; CHECK:        i8 MemRef_Short[*]; // Element size 1
; CHECK:        i16 MemRef_Float[*]; // Element size 2
; CHECK:        i8 MemRef_Double[*]; // Element size 1
; CHECK:    }
;
; CHECK: Statements {
; CHECK-NEXT: Stmt_bb2
; CHECK-NEXT: Domain :=
; CHECK-NEXT:     { Stmt_bb2[i0] : 0 <= i0 <= 99 };
; CHECK-NEXT: Schedule :=
; CHECK-NEXT:     { Stmt_bb2[i0] -> [i0] };
; CHECK-NEXT: ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:     { Stmt_bb2[i0] -> MemRef_Short[o0] : 0 <= o0 <= 9 and ((o0 >= 8 and 16*floor((8 + i0)/16) > i0) or (o0 <= 1 and 16*floor((8 + i0)/16) <= i0)) };
; CHECK-NEXT: MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:     { Stmt_bb2[i0] -> MemRef_Short[i0] };
; CHECK-NEXT: ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:     { Stmt_bb2[i0] -> MemRef_Float[o0] : 0 <= o0 <= 9 and ((o0 >= 8 and 16*floor((8 + i0)/16) > i0) or (o0 <= 1 and 16*floor((8 + i0)/16) <= i0)) };
; CHECK-NEXT: MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:     { Stmt_bb2[i0] -> MemRef_Float[i0] };
; CHECK-NEXT: ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:     { Stmt_bb2[i0] -> MemRef_Double[o0] : 0 <= o0 <= 15 and ((o0 >= 8 and 16*floor((8 + i0)/16) > i0) or (o0 <= 7 and 16*floor((8 + i0)/16) <= i0)) };
; CHECK-NEXT: MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:     { Stmt_bb2[i0] -> MemRef_Double[i0] };
; CHECK-NEXT: }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @multiple_types(i8* noalias %Short, i16* noalias %Float, i8* noalias %Double) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb20, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp21, %bb20 ]
  %exitcond = icmp ne i64 %i.0, 100
  br i1 %exitcond, label %bb2, label %bb22

bb2:                                              ; preds = %bb1
  %quad = and i64 %i.0, 8
  %tmp3 = getelementptr inbounds i8, i8* %Short, i64 %quad
  %tmp4 = bitcast i8* %tmp3 to i16*
  %tmp5 = load i16, i16* %tmp4, align 2
  %tmp6 = trunc i16 %tmp5 to i8
  %tmp7 = getelementptr inbounds i8, i8* %Short, i64 %i.0
  store i8 %tmp6, i8* %tmp7, align 1
  %tmp9 = getelementptr inbounds i16, i16* %Float, i64 %quad
  %tmp10 = bitcast i16* %tmp9 to float*
  %tmp11 = load float, float* %tmp10, align 4
  %tmp12 = fptosi float %tmp11 to i16
  %tmp13 = getelementptr inbounds i16, i16* %Float, i64 %i.0
  store i16 %tmp12, i16* %tmp13, align 1
  %tmp15 = getelementptr inbounds i8, i8* %Double, i64 %quad
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
