; RUN: opt %loadPolly -polly-print-scops -disable-output < %s 2>&1 | not FileCheck %s

;#define TYPE float
;#define NUM 4
;
;TYPE A[NUM];
;TYPE B[NUM];
;TYPE C[NUM];
;
;void vector_multiply(void) {
;	int i;
;	for (i = 0; i < NUM; i++) {
;		A[i] = B[i] * C[i];
;	}
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

@B = common global [4 x float] zeroinitializer, align 16
@C = common global [4 x float] zeroinitializer, align 16
@A = common global [4 x float] zeroinitializer, align 16

define void @vector_multiply() nounwind {
bb:
  br label %bb3

bb3:                                              ; preds = %bb7, %bb
  %indvar = phi i64 [ %indvar.next, %bb7 ], [ 0, %bb ]
  %scevgep = getelementptr [4 x float], ptr @A, i64 0, i64 %indvar
  %scevgep1 = getelementptr [4 x float], ptr @C, i64 0, i64 %indvar
  %scevgep2 = getelementptr [4 x float], ptr @B, i64 0, i64 %indvar
  %exitcond = icmp ne i64 %indvar, 4
  br i1 %exitcond, label %bb4, label %bb8

bb4:                                              ; preds = %bb3
  %tmp = load float, ptr %scevgep2, align 4
  %tmp5 = load float, ptr %scevgep1, align 4
  %tmp6 = fmul float %tmp, %tmp5
  store float %tmp6, ptr %scevgep, align 4
  br label %bb7

bb7:                                              ; preds = %bb4
  %indvar.next = add i64 %indvar, 1
  br label %bb3

bb8:                                              ; preds = %bb3
  ret void
}

; Match any reduction type except "[Reduction Type: NONE]"
; CHECK:     [Reduction Type: {{[^N].*}}]
