; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnueabi"

; Make sure we don't crash trying delinearize the memory access
; CHECK: region: 'bb4 => bb14'
; CHECK-NEXT: Invalid Scop!
define void @f(ptr %arg, i32 %arg1, i32 %arg2, i32 %arg3) {
bb:
  br label %bb4

bb4:
  %tmp = phi i32 [ %arg2, %bb ], [ %tmp12, %bb4 ]
  %tmp5 = icmp sgt i32 %tmp, 0
  %tmp6 = select i1 %tmp5, i32 %tmp, i32 0
  %tmp7 = mul nsw i32 %tmp6, %arg3
  %tmp8 = sext i32 %tmp7 to i64
  %tmp9 = getelementptr inbounds i8, ptr %arg, i64 %tmp8
  %tmp11 = load i32, ptr %tmp9, align 4
  %tmp12 = add nsw i32 %tmp, 1
  %tmp13 = icmp slt i32 %tmp, %arg1
  br i1 %tmp13, label %bb4, label %bb14

bb14:
  ret void
}
