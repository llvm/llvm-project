; RUN: opt < %s -force-vector-width=2 -force-vector-interleave=2 -passes=loop-vectorize -S | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"

; CHECK-LABEL: @dead_instructions_01
;
; This test ensures that we don't generate trivially dead instructions prior to
; instruction simplification. We don't need to generate instructions
; corresponding to the original induction variable update or branch condition,
; since we rewrite the loop structure.
;
; CHECK:     vector.body:
; CHECK:       %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK:       %[[I0:.+]] = add i64 %index, 0
; CHECK:       %[[I2:.+]] = add i64 %index, 2
; CHECK:       getelementptr inbounds i64, i64* %a, i64 %[[I0]]
; CHECK:       getelementptr inbounds i64, i64* %a, i64 %[[I2]]
; CHECK-NOT:   add nuw nsw i64 %[[I0]], 1
; CHECK-NOT:   add nuw nsw i64 %[[I2]], 1
; CHECK-NOT:   icmp slt i64 {{.*}}, %n
; CHECK:       %index.next = add nuw i64 %index, 4
; CHECK:       %[[CMP:.+]] = icmp eq i64 %index.next, %n.vec
; CHECK:       br i1 %[[CMP]], label %middle.block, label %vector.body
;
define i64 @dead_instructions_01(i64 *%a, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %r = phi i64 [ %tmp2, %for.body ], [ 0, %entry ]
  %tmp0 = getelementptr inbounds i64, i64* %a, i64 %i
  %tmp1 = load i64, i64* %tmp0, align 8
  %tmp2 = add i64 %tmp1, %r
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  %tmp3  = phi i64 [ %tmp2, %for.body ]
  ret i64 %tmp3
}


; CHECK-LABEL: @pr47390
;
; This test ensures that the primary induction is not considered dead when
; acting as the 'add' of another induction, and otherwise feeding only its own
; 'add' (recognized earlier as 'dead'), when the tail of the loop is folded by
; masking. Such masking uses the primary induction.
;
; CHECK:     vector.body:
;
define void @pr47390(i32 *%a) {
entry:
  br label %loop

exit:
  ret void

loop:
  %primary = phi i32 [ 0, %entry ], [ %primary_add, %loop ]
  %use_primary = phi i32 [ -1, %entry ], [ %primary, %loop ]
  %secondary = phi i32 [ 1, %entry ], [ %secondary_add, %loop ]
  %primary_add = add i32 %primary, 1
  %secondary_add = add i32 %secondary, 1
  %gep = getelementptr inbounds i32, i32* %a, i32 %secondary
  %load = load i32, i32* %gep, align 8
  %cmp = icmp eq i32 %secondary, 5
  br i1 %cmp, label %exit, label %loop
}
