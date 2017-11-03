; RUN: opt -S -loop-predication -loop-predication-enable-iv-truncation=true < %s 2>&1 | FileCheck %s
declare void @llvm.experimental.guard(i1, ...)

declare i32 @length(i8*)

declare i16 @short_length(i8*)
; Consider range check of type i16 and i32, while IV is of type i64
; We can loop predicate this because the IV range is within i16 and within i32.
define i64 @iv_wider_type_rc_two_narrow_types(i32 %offA, i16 %offB, i8* %arrA, i8* %arrB) {
; CHECK-LABEL: iv_wider_type_rc_two_narrow_types
entry:
; CHECK-LABEL: entry:
; CHECK: [[idxB:[^ ]+]] = sub i16 %lengthB, %offB
; CHECK-NEXT: [[limit_checkB:[^ ]+]] = icmp ule i16 16, [[idxB]]
; CHECK-NEXT: [[first_iteration_checkB:[^ ]+]] = icmp ult i16 %offB, %lengthB
; CHECK-NEXT: [[WideChkB:[^ ]+]] = and i1 [[first_iteration_checkB]], [[limit_checkB]]
; CHECK-NEXT: [[idxA:[^ ]+]] = sub i32 %lengthA, %offA
; CHECK-NEXT: [[limit_checkA:[^ ]+]] = icmp ule i32 16, [[idxA]]
; CHECK-NEXT: [[first_iteration_checkA:[^ ]+]] = icmp ult i32 %offA, %lengthA
; CHECK-NEXT: [[WideChkA:[^ ]+]] = and i1 [[first_iteration_checkA]], [[limit_checkA]]
  %lengthA = call i32 @length(i8* %arrA)
  %lengthB = call i16 @short_length(i8* %arrB)
   br label %loop

loop:
; CHECK-LABEL: loop:
; CHECK: [[invariant_check:[^ ]+]] = and i1 [[WideChkB]], [[WideChkA]]
; CHECK-NEXT: call void (i1, ...) @llvm.experimental.guard(i1 [[invariant_check]], i32 9)
  %iv = phi i64 [0, %entry ], [ %iv.next, %loop ]
  %iv.trunc.32 = trunc i64 %iv to i32
  %iv.trunc.16 = trunc i64 %iv to i16
  %indexA = add i32 %iv.trunc.32, %offA
  %indexB = add i16 %iv.trunc.16, %offB
  %rcA = icmp ult i32 %indexA, %lengthA
  %rcB = icmp ult i16 %indexB, %lengthB
  %wide.chk = and i1 %rcA, %rcB
  call void (i1, ...) @llvm.experimental.guard(i1 %wide.chk, i32 9) [ "deopt"() ]
  %indexA.ext = zext i32 %indexA to i64
  %addrA = getelementptr inbounds i8, i8* %arrA, i64 %indexA.ext
  %eltA = load i8, i8* %addrA
  %indexB.ext = zext i16 %indexB to i64
  %addrB = getelementptr inbounds i8, i8* %arrB, i64 %indexB.ext
  store i8 %eltA, i8* %addrB
  %iv.next = add nuw nsw i64 %iv, 1
  %latch.check = icmp ult i64 %iv.next, 16
  br i1 %latch.check, label %loop, label %exit

exit:
 ret i64 %iv
}


; Consider an IV of type long and an array access into int array.
; IV is of type i64 while the range check operands are of type i32 and i64.
define i64 @iv_rc_different_types(i32 %offA, i32 %offB, i8* %arrA, i8* %arrB, i64 %max)
{
; CHECK-LABEL: iv_rc_different_types
entry:
; CHECK-LABEL: entry:
; CHECK: [[lenB:[^ ]+]] = add i32 %lengthB, -1
; CHECK-NEXT: [[idxB:[^ ]+]] = sub i32 [[lenB]], %offB
; CHECK-NEXT: [[limit_checkB:[^ ]+]] = icmp ule i32 15, [[idxB]]
; CHECK-NEXT: [[first_iteration_checkB:[^ ]+]] = icmp ult i32 %offB, %lengthB
; CHECK-NEXT: [[WideChkB:[^ ]+]] = and i1 [[first_iteration_checkB]], [[limit_checkB]]
; CHECK-NEXT: [[maxMinusOne:[^ ]+]] = add i64 %max, -1
; CHECK-NEXT: [[limit_checkMax:[^ ]+]] = icmp ule i64 15, [[maxMinusOne]]
; CHECK-NEXT: [[first_iteration_checkMax:[^ ]+]] = icmp ult  i64 0, %max
; CHECK-NEXT: [[WideChkMax:[^ ]+]] = and i1 [[first_iteration_checkMax]], [[limit_checkMax]]
; CHECK-NEXT: [[lenA:[^ ]+]] = add i32 %lengthA, -1
; CHECK-NEXT: [[idxA:[^ ]+]] = sub i32 [[lenA]], %offA
; CHECK-NEXT: [[limit_checkA:[^ ]+]] = icmp ule i32 15, [[idxA]]
; CHECK-NEXT: [[first_iteration_checkA:[^ ]+]] = icmp ult i32 %offA, %lengthA
; CHECK-NEXT: [[WideChkA:[^ ]+]] = and i1 [[first_iteration_checkA]], [[limit_checkA]]
  %lengthA = call i32 @length(i8* %arrA)
  %lengthB = call i32 @length(i8* %arrB)
  br label %loop

loop:
; CHECK-LABEL: loop:
; CHECK: [[BandMax:[^ ]+]] = and i1 [[WideChkB]], [[WideChkMax]]
; CHECK: [[ABandMax:[^ ]+]] = and i1 [[BandMax]], [[WideChkA]]
; CHECK: call void (i1, ...) @llvm.experimental.guard(i1 [[ABandMax]], i32 9)
  %iv = phi i64 [0, %entry ], [ %iv.next, %loop ]
  %iv.trunc = trunc i64 %iv to i32
  %indexA = add i32 %iv.trunc, %offA
  %indexB = add i32 %iv.trunc, %offB
  %rcA = icmp ult i32 %indexA, %lengthA
  %rcIV = icmp ult i64 %iv, %max
  %wide.chk = and i1 %rcA, %rcIV
  %rcB = icmp ult i32 %indexB, %lengthB
  %wide.chk.final = and i1 %wide.chk, %rcB
  call void (i1, ...) @llvm.experimental.guard(i1 %wide.chk.final, i32 9) [ "deopt"() ]
  %indexA.ext = zext i32 %indexA to i64
  %addrA = getelementptr inbounds i8, i8* %arrA, i64 %indexA.ext
  %eltA = load i8, i8* %addrA
  %indexB.ext = zext i32 %indexB to i64
  %addrB = getelementptr inbounds i8, i8* %arrB, i64 %indexB.ext
  %eltB = load i8, i8* %addrB
  %result = xor i8 %eltA, %eltB
  store i8 %result, i8* %addrA
  %iv.next = add nuw nsw i64 %iv, 1
  %latch.check = icmp ult i64 %iv, 15
  br i1 %latch.check, label %loop, label %exit

exit:
  ret i64 %iv
}

; cannot narrow the IV to the range type, because we lose information.
; for (i64 i= 5; i>= 2; i++)
; this loop wraps around after reaching 2^64.
define i64 @iv_rc_different_type(i32 %offA, i8* %arrA) {
; CHECK-LABEL: iv_rc_different_type
entry:
  %lengthA = call i32 @length(i8* %arrA)
  br label %loop

loop:
; CHECK-LABEL: loop:
; CHECK: %rcA = icmp ult i32 %indexA, %lengthA
; CHECK-NEXT: call void (i1, ...) @llvm.experimental.guard(i1 %rcA, i32 9)
  %iv = phi i64 [ 5, %entry ], [ %iv.next, %loop ]
  %iv.trunc.32 = trunc i64 %iv to i32
  %indexA = add i32 %iv.trunc.32, %offA
  %rcA = icmp ult i32 %indexA, %lengthA
  call void (i1, ...) @llvm.experimental.guard(i1 %rcA, i32 9) [ "deopt"() ]
  %indexA.ext = zext i32 %indexA to i64
  %addrA = getelementptr inbounds i8, i8* %arrA, i64 %indexA.ext
  %eltA = load i8, i8* %addrA
  %res = add i8 %eltA, 2
  store i8 %eltA, i8* %addrA
  %iv.next = add i64 %iv, 1
  %latch.check = icmp sge i64 %iv.next, 2
  br i1 %latch.check, label %loop, label %exit

exit:
 ret i64 %iv
}
