; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-code \
; RUN: -disable-output < %s | \
; RUN: FileCheck -check-prefix=CODE %s
;
; REQUIRES: pollyacc
;
;    void foo(float A[], float B[]) {
;      for (long i = 0; i < 1024; i++)
;        A[2 * i] = B[i];
;    }

; CODE: cudaCheckReturn(cudaMemcpy(dev_MemRef_B, MemRef_B, (1024) * sizeof(i32), cudaMemcpyHostToDevice));
; CODE-NEXT: cudaCheckReturn(cudaMemcpy(dev_MemRef_A, MemRef_A, (2047) * sizeof(i32), cudaMemcpyHostToDevice));

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(ptr %A, ptr %B) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb8, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp9, %bb8 ]
  %exitcond = icmp ne i64 %i.0, 1024
  br i1 %exitcond, label %bb2, label %bb10

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds float, ptr %B, i64 %i.0
  %tmp4 = load i32, ptr %tmp, align 4
  %tmp5 = shl nsw i64 %i.0, 1
  %tmp6 = getelementptr inbounds float, ptr %A, i64 %tmp5
  store i32 %tmp4, ptr %tmp6, align 4
  br label %bb8

bb8:                                              ; preds = %bb2
  %tmp9 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb10:                                             ; preds = %bb1
  ret void
}
