; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -S %s | FileCheck --check-prefix=VF4 %s
; RUN: opt -passes=loop-vectorize -force-vector-interleave=2 -force-vector-width=1 -S %s | FileCheck --check-prefix=IC2 %s

; rdar://problem/12848162

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

@a = common global [2048 x i32] zeroinitializer, align 16

define void @example12() {
; VF4-LABEL: @example12(
; VF4-LABEL: vector.body:
; VF4: [[VEC_IND:%.+]] = phi <4 x i32>
; VF4: store <4 x i32> [[VEC_IND]]
; VF4: middle.block:
;
; IC2-LABEL: @example12(
; IC2-LABEL: vector.body:
; IC2-NEXT:   [[INDEX:%.+]] = phi i64 [ 0, %vector.ph ]
; IC2-NEXT:   [[TRUNC:%.+]] = trunc i64 [[INDEX]] to i32
; IC2-NEXT:   [[TRUNC0:%.+]] = add i32 [[TRUNC]], 0
; IC2-NEXT:   [[TRUNC1:%.+]] = add i32 [[TRUNC]], 1
; IC2:        store i32 [[TRUNC0]],
; IC2-NEXT:   store i32 [[TRUNC1]],
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep = getelementptr inbounds [2048 x i32], [2048 x i32]* @a, i64 0, i64 %iv
  %iv.trunc = trunc i64 %iv to i32
  store i32 %iv.trunc, i32* %gep, align 4
  %iv.next = add i64 %iv, 1
  %iv.next.trunc = trunc i64 %iv.next to i32
  %exitcond = icmp eq i32 %iv.next.trunc, 1024
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

