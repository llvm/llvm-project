; RUN: opt %s -passes=loop-vectorize -force-vector-interleave=2 -force-vector-width=4 -S | FileCheck %s

; Fixes PR43828

define void @test(ptr %B) {
; CHECK-LABEL: @test(
; CHECK:       vector.body:
; CHECK-COUNT-2: sub <4 x i32>
entry:
  br label %outer_loop

outer_loop:
  %local_4 = phi i32 [ 2, %entry ], [ %4, %outer_tail]
  br label %inner_loop

inner_loop:
  %local_2 = phi i32 [ 0, %outer_loop ], [ %1, %inner_loop ]
  %local_3 = phi i32 [ -104, %outer_loop ], [ %0, %inner_loop ]
  %0 = sub nuw nsw i32 %local_3, %local_4
  %1 = add nuw nsw i32 %local_2, 1
  %2 = icmp ugt i32 %local_2, 126
  br i1 %2, label %outer_tail, label %inner_loop

outer_tail:
  %3 = phi i32 [ %0, %inner_loop ]
  store atomic i32 %3, ptr %B unordered, align 8
  %4 = add i32 %local_4, 1
  %5 = icmp slt i32 %4, 6
  br i1 %5, label %outer_loop, label %exit

exit:
  ret void
}

define i32 @multi-instr(ptr noalias nocapture %A, ptr noalias nocapture %B, i32 %inc) {
; CHECK-LABEL: @multi-instr(
; CHECK:       vector.body:
; CHECK-COUNT-4: add <4 x i32>
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv_inc, %loop]
  %redu = phi i32 [0, %entry], [%3, %loop]
  %gepa = getelementptr inbounds i32, ptr %A, i32 %iv
  %gepb = getelementptr inbounds i32, ptr %B, i32 %iv
  %0 = load i32, ptr %gepa
  %1 = load i32, ptr %gepb
  %2 = add nuw nsw i32 %redu, %0
  %3 = add nuw nsw i32 %2, %1
  %iv_inc = add nuw nsw i32 %iv, 1
  %4 = icmp ult i32 %iv_inc, 128
  br i1 %4, label %loop, label %exit

exit:
  %lcssa = phi i32 [%3, %loop]
  ret i32 %lcssa
}
