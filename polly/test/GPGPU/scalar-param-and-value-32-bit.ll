; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-kernel-ir \
; RUN: -disable-output < %s | \
; RUN: FileCheck %s

; REQUIRES: pollyacc, target=nvptx{{.*}}
;
;    void foo(float A[], int n) {
;      for (long j = 0; j < n; j++)
;        A[j + n] += 42;
;    }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; CHECK: define ptx_kernel void @FUNC_foo_SCOP_0_KERNEL_0(ptr addrspace(1) %MemRef_A, i32 %n)

define void @foo(ptr %A, i32 %n) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb9, %bb
  %j.0 = phi i64 [ 0, %bb ], [ %tmp10, %bb9 ]
  %tmp = sext i32 %n to i64
  %tmp2 = icmp slt i64 %j.0, %tmp
  br i1 %tmp2, label %bb3, label %bb11

bb3:                                              ; preds = %bb1
  %tmp4 = sext i32 %n to i64
  %tmp5 = add nsw i64 %j.0, %tmp4
  %tmp6 = getelementptr inbounds float, ptr %A, i64 %tmp5
  %tmp7 = load float, ptr %tmp6, align 4
  %tmp8 = fadd float %tmp7, 4.200000e+01
  store float %tmp8, ptr %tmp6, align 4
  br label %bb9

bb9:                                              ; preds = %bb3
  %tmp10 = add nuw nsw i64 %j.0, 1
  br label %bb1

bb11:                                             ; preds = %bb1
  ret void
}
