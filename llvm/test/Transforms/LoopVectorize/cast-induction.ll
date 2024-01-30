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
  %gep = getelementptr inbounds [2048 x i32], ptr @a, i64 0, i64 %iv
  %iv.trunc = trunc i64 %iv to i32
  store i32 %iv.trunc, ptr %gep, align 4
  %iv.next = add i64 %iv, 1
  %iv.next.trunc = trunc i64 %iv.next to i32
  %exitcond = icmp eq i32 %iv.next.trunc, 1024
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

define void @redundant_iv_cast(ptr %dst) {
; VF4-LABEL: @redundant_iv_cast
; VF4: vector.body:
; VF4:   [[VEC_IND:%.+]] = phi <4 x i16> [ <i16 0, i16 1, i16 2, i16 3>, %vector.ph ], [ [[VEC_IND_NEXT:%.+]], %vector.body ]
; VF4:  store <4 x i16> [[VEC_IND]]
; VF4:  [[VEC_IND_NEXT]] = add <4 x i16> [[VEC_IND]], <i16 4, i16 4, i16 4, i16 4>
;
; IC2-LABEL: @redundant_iv_cast
; IC2:      vector.body:
; IC2-NEXT:  [[CAN_IV:%.+]] = phi i32 [ 0, %vector.ph ], [ [[CAN_IV_NEXT:%.+]], %vector.body ]
; IC2-NEXT:  [[OFFSET_IDX:%.+]] = trunc i32 [[CAN_IV]] to i16
; IC2-NEXT:  [[P0:%.+]] = add i16 [[OFFSET_IDX]], 0
; IC2-NEXT:  [[P1:%.+]] = add i16 [[OFFSET_IDX]], 1
; IC2:       store i16 [[P0]]
; IC2-NEXT:  store i16 [[P1]]
;
entry:
  br label %loop

loop:
  %j.0 = phi i16 [ 0, %entry ], [ %inc, %loop ]
  %ext = zext i16 %j.0 to i32
  %trunc = trunc i32 %ext to i16
  %gep = getelementptr inbounds i16, ptr %dst, i16 %j.0
  store i16 %trunc, ptr %gep
  %0 = icmp eq i16 10000, %j.0
  %inc = add i16 %j.0, 1
  br i1 %0, label %exit, label %loop


exit:
  ret void
}


define void @cast_variable_step(i64 %step) {
; VF4-LABEL: @cast_variable_step(
; VF4-LABEL: vector.body:
; VF4: [[VEC_IND:%.+]] = phi <4 x i32>
; VF4: store <4 x i32> [[VEC_IND]]
; VF4: middle.block:
;
; IC2-LABEL: @cast_variable_step(
; IC2:   [[TRUNC_STEP:%.+]] = trunc i64 %step to i32
; IC2:   br label %vector.body

; IC2-LABEL: vector.body:
; IC2-NEXT:   [[INDEX:%.+]] = phi i64 [ 0, %vector.ph ]
; IC2-NEXT:   [[MUL:%.+]] = mul i64 %index, %step
; IC2-NEXT:   [[OFFSET_IDX:%.+]] = add i64 10, [[MUL]]
; IC2-NEXT:   [[TRUNC_OFF:%.+]] = trunc i64 [[OFFSET_IDX]] to i32
; IC2-NEXT:   [[STEP0:%.+]] = mul i32 0, [[TRUNC_STEP]]
; IC2-NEXT:   [[T0:%.+]] = add i32 [[TRUNC_OFF]], [[STEP0]]
; IC2-NEXT:   [[STEP1:%.+]] = mul i32 1, [[TRUNC_STEP]]
; IC2-NEXT:   [[T1:%.+]] = add i32 [[TRUNC_OFF]], [[STEP1]]
; IC2:        store i32 [[T0]],
; IC2-NEXT:   store i32 [[T1]],
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.2 = phi i64 [ 10, %entry ], [ %iv.2.next, %loop ]
  %gep = getelementptr inbounds [2048 x i32], ptr @a, i64 0, i64 %iv
  %iv.2.trunc = trunc i64 %iv.2 to i32
  store i32 %iv.2.trunc, ptr %gep, align 4
  %iv.next = add i64 %iv, 1
  %iv.2.next = add i64 %iv.2, %step
  %exitcond = icmp eq i64 %iv.next, 1024
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}
