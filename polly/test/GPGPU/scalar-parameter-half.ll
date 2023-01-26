; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-code -disable-output %s

; REQUIRES: pollyacc, target=nvptx{{.*}}

;    void foo(half A[], half b) {
;      for (long i = 0; i < 1024; i++)
;        A[i] += b;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @half(ptr %A, half %b) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb5, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp6, %bb5 ]
  %exitcond = icmp ne i64 %i.0, 1024
  br i1 %exitcond, label %bb2, label %bb7

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds half, ptr %A, i64 %i.0
  %tmp3 = load half, ptr %tmp, align 4
  %tmp4 = fadd half %tmp3, %b
  store half %tmp4, ptr %tmp, align 4
  br label %bb5

bb5:                                              ; preds = %bb2
  %tmp6 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb7:                                              ; preds = %bb1
  ret void
}

