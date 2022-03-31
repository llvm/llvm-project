; RUN: opt -S -disable-output -passes='print-access-info' %s 2>&1 | FileCheck %s

;
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; A forwarding in the presence of symbolic strides.
define void @single_stride(i32* noalias %A, i32* noalias %B, i64 %N, i64 %stride) {
; CHECK-LABEL: Loop access info in function 'single_stride':
; CHECK-NEXT:  loop:
; CHECK-NEXT:    Report: unsafe dependent memory operations in loop.
; CHECK-NEXT:    Backward loop carried data dependence.
; CHECK-NEXT:    Dependences:
; CHECK-NEXT:      Backward:
; CHECK-NEXT:          %load = load i32, i32* %gep.A, align 4 ->
; CHECK-NEXT:          store i32 %add, i32* %gep.A.next, align 4
; CHECK-EMPTY:
; CHECK-NEXT:    Run-time memory checks:
; CHECK-NEXT:    Grouped accesses:
; CHECK-EMPTY:
; CHECK-NEXT:    Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:    SCEV assumptions:
; CHECK-NEXT:    Equal predicate: %stride == 1
; CHECK-EMPTY:
; CHECK-NEXT:    Expressions re-written:
; CHECK-NEXT:    [PSE]  %gep.A = getelementptr inbounds i32, i32* %A, i64 %mul:
; CHECK-NEXT:      {%A,+,(4 * %stride)}<%loop>
; CHECK-NEXT:      --> {%A,+,4}<%loop>
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %mul = mul i64 %iv, %stride
  %gep.A = getelementptr inbounds i32, i32* %A, i64 %mul
  %load = load i32, i32* %gep.A, align 4
  %gep.B = getelementptr inbounds i32, i32* %B, i64 %iv
  %load_1 = load i32, i32* %gep.B, align 4
  %add = add i32 %load_1, %load
  %iv.next = add nuw nsw i64 %iv, 1
  %gep.A.next = getelementptr inbounds i32, i32* %A, i64 %iv.next
  store i32 %add, i32* %gep.A.next, align 4
  %exitcond = icmp eq i64 %iv.next, %N
  br i1 %exitcond, label %exit, label %loop

exit:                                          ; preds = %loop
  ret void
}

; Similar to @single_stride, but with struct types.
define void @single_stride_struct({ i32, i8 }* noalias %A, { i32, i8 }* noalias %B, i64 %N, i64 %stride) {
; CHECK-LABEL: Loop access info in function 'single_stride_struct':
; CHECK-NEXT:  loop:
; CHECK-NEXT:    Report: unsafe dependent memory operations in loop.
; CHECK-NEXT:    Backward loop carried data dependence.
; CHECK-NEXT:    Dependences:
; CHECK-NEXT:      Backward:
; CHECK-NEXT:          %load = load { i32, i8 }, { i32, i8 }* %gep.A, align 4 ->
; CHECK-NEXT:          store { i32, i8 } %ins, { i32, i8 }* %gep.A.next, align 4
; CHECK-EMPTY:
; CHECK-NEXT:    Run-time memory checks:
; CHECK-NEXT:    Grouped accesses:
; CHECK-EMPTY:
; CHECK-NEXT:    Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:    SCEV assumptions:
; CHECK-NEXT:    Equal predicate: %stride == 1
; CHECK-EMPTY:
; CHECK-NEXT:    Expressions re-written:
; CHECK-NEXT:    [PSE]  %gep.A = getelementptr inbounds { i32, i8 }, { i32, i8 }* %A, i64 %mul:
; CHECK-NEXT:      {%A,+,(8 * %stride)}<%loop>
; CHECK-NEXT:      --> {%A,+,8}<%loop>
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %mul = mul i64 %iv, %stride
  %gep.A = getelementptr inbounds { i32, i8 }, { i32, i8 }* %A, i64 %mul
  %load = load { i32, i8 }, { i32, i8 }* %gep.A, align 4
  %gep.B = getelementptr inbounds { i32, i8 }, { i32, i8 }* %B, i64 %iv
  %load_1 = load { i32, i8 }, { i32, i8 }* %gep.B, align 4
  %v1 = extractvalue { i32, i8 } %load, 0
  %v2 = extractvalue { i32, i8} %load_1, 0
  %add = add i32 %v1, %v2
  %ins = insertvalue { i32, i8 } undef, i32 %add, 0
  %iv.next = add nuw nsw i64 %iv, 1
  %gep.A.next = getelementptr inbounds { i32, i8 }, { i32, i8 }* %A, i64 %iv.next
  store { i32, i8 } %ins, { i32, i8 }* %gep.A.next, align 4
  %exitcond = icmp eq i64 %iv.next, %N
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

; A loop with two symbolic strides.
define void @two_strides(i32* noalias %A, i32* noalias %B, i64 %N, i64 %stride.1, i64 %stride.2) {
; CHECK-LABEL: Loop access info in function 'two_strides':
; CHECK-NEXT:  loop:
; CHECK-NEXT:    Report: unsafe dependent memory operations in loop.
; CHECK-NEXT:    Backward loop carried data dependence.
; CHECK-NEXT:    Dependences:
; CHECK-NEXT:      Backward:
; CHECK-NEXT:          %load = load i32, i32* %gep.A, align 4 ->
; CHECK-NEXT:          store i32 %add, i32* %gep.A.next, align 4
; CHECK-EMPTY:
; CHECK-NEXT:    Run-time memory checks:
; CHECK-NEXT:    Grouped accesses:
; CHECK-EMPTY:
; CHECK-NEXT:    Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:    SCEV assumptions:
; CHECK-NEXT:    Equal predicate: %stride.2 == 1
; CHECK-NEXT:    Equal predicate: %stride.1 == 1
; CHECK-EMPTY:
; CHECK-NEXT:    Expressions re-written:
; CHECK-NEXT:    [PSE]  %gep.A = getelementptr inbounds i32, i32* %A, i64 %mul:
; CHECK-NEXT:      {%A,+,(4 * %stride.1)}<%loop>
; CHECK-NEXT:      --> {%A,+,4}<%loop>
; CHECK-NEXT:    [PSE]  %gep.A.next = getelementptr inbounds i32, i32* %A, i64 %mul.2:
; CHECK-NEXT:      {((4 * %stride.2) + %A),+,(4 * %stride.2)}<%loop>
; CHECK-NEXT:      --> {(4 + %A),+,4}<%loop>
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %mul = mul i64 %iv, %stride.1
  %gep.A = getelementptr inbounds i32, i32* %A, i64 %mul
  %load = load i32, i32* %gep.A, align 4
  %gep.B = getelementptr inbounds i32, i32* %B, i64 %iv
  %load_1 = load i32, i32* %gep.B, align 4
  %add = add i32 %load_1, %load
  %iv.next = add nuw nsw i64 %iv, 1
  %mul.2 = mul i64 %iv.next, %stride.2
  %gep.A.next = getelementptr inbounds i32, i32* %A, i64 %mul.2
  store i32 %add, i32* %gep.A.next, align 4
  %exitcond = icmp eq i64 %iv.next, %N
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}
