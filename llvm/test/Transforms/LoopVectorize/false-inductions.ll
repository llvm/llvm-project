; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize -S -disable-output %s 2>&1 | FileCheck %s
; REQUIRES: asserts

define i32 @test_var_addend(i32 %addend, ptr %p) {
; CHECK: LV: Not vectorizing: Found an unidentified PHI
entry:
  br label %loop

loop:
  %iv0 = phi i32 [ 1, %entry ], [ %iv0.next, %loop ]
  %iv1 = phi i32 [ 0, %entry ], [ %iv1.next, %loop ]
  %iv2 = phi i32 [ 0, %entry ], [ %iv2.next, %loop ]
  %iv0.next = add nuw nsw i32 %iv0, 1
  %cmp = icmp ult i32 %iv0, 2
  %select = select i1 %cmp, i32 %addend, i32 0
  %iv1.next = add i32 %select, %iv1
  %iv2.next = add i32 %iv2, %iv1.next
  br i1 %cmp, label %loop, label %exit

exit:
  store atomic i32 %iv2.next, ptr %p unordered, align 8
  ret i32 %iv1.next
}

define i32 @test_var_single_step(ptr %p) {
; CHECK: LV: Not vectorizing: Found an unidentified PHI
entry:
  br label %loop

loop:
  %iv0 = phi i32 [ 1, %entry ], [ %iv0.next, %loop ]
  %iv1 = phi i32 [ 0, %entry ], [ %iv1.next, %loop ]
  %iv2 = phi i32 [ 0, %entry ], [ %iv2.next, %loop ]
  %iv0.next = add nuw nsw i32 %iv0, 1
  %cmp = icmp ult i32 %iv0, 2
  %select = select i1 %cmp, i32 1, i32 0
  %iv1.next = add i32 %select, %iv1
  %iv2.next = add i32 %iv2, %iv1.next
  br i1 %cmp, label %loop, label %exit

exit:
  store atomic i32 %iv2.next, ptr %p unordered, align 8
  ret i32 %iv1.next
}
