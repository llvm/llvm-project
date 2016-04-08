; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
; CHECK:    Invalid Context:
; CHECK:        [N] -> {  : N >= 129 }
;
;    void foo(float *A, long N) {
;      for (long i = 0; i < N; i++)
;        if ((signed char)i < 100)
;          A[i] += i;
;    }
define void @foo(float* %A, i64 %N) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb11, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp12, %bb11 ]
  %tmp = icmp slt i64 %i.0, %N
  br i1 %tmp, label %bb2, label %bb13

bb2:                                              ; preds = %bb1
  %tmp3 = trunc i64 %i.0 to i8
  %tmp4 = icmp slt i8 %tmp3, 100
  br i1 %tmp4, label %bb5, label %bb10

bb5:                                              ; preds = %bb2
  %tmp6 = sitofp i64 %i.0 to float
  %tmp7 = getelementptr inbounds float, float* %A, i64 %i.0
  %tmp8 = load float, float* %tmp7, align 4
  %tmp9 = fadd float %tmp8, %tmp6
  store float %tmp9, float* %tmp7, align 4
  br label %bb10

bb10:                                             ; preds = %bb5, %bb2
  br label %bb11

bb11:                                             ; preds = %bb10
  %tmp12 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb13:                                             ; preds = %bb1
  ret void
}
