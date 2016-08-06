; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-kernel-ir \
; RUN: -disable-output < %s | \
; RUN: FileCheck -check-prefix=KERNEL %s

; REQUIRES: pollyacc

; KERNEL: define ptx_kernel void @kernel_0(i8* %MemRef_A, i64 %n) #0 {

; KERNEL: !nvvm.annotations = !{!0}

; KERNEL: !0 = !{void (i8*, i64)* @kernel_0, !"maxntidx", i32 32, !"maxntidy", i32 1, !"maxntidz", i32 1}

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(i64* %A, i64 %n) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb6, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp7, %bb6 ]
  %tmp = icmp slt i64 %i.0, %n
  br i1 %tmp, label %bb2, label %bb8

bb2:                                              ; preds = %bb1
  %tmp3 = getelementptr inbounds i64, i64* %A, i64 %i.0
  %tmp4 = load i64, i64* %tmp3, align 8
  %tmp5 = add nsw i64 %tmp4, 100
  store i64 %tmp5, i64* %tmp3, align 8
  br label %bb6

bb6:                                              ; preds = %bb2
  %tmp7 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb8:                                              ; preds = %bb1
  ret void
}
