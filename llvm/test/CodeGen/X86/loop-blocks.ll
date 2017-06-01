; RUN: llc < %s -march=x86-64 -mtriple=x86_64-unknown-linux-gnu -asm-verbose=false | FileCheck %s

; These tests check for loop branching structure, and that the loop align
; directive is placed in the expected place.

; CodeGen should insert a branch into the middle of the loop in
; order to avoid a branch within the loop.

; CHECK-LABEL: simple:
;      CHECK:   jmp   .LBB0_1
; CHECK-NEXT:   align
; CHECK-NEXT: .LBB0_2:
; CHECK-NEXT:   callq loop_latch
; CHECK-NEXT: .LBB0_1:
; CHECK-NEXT:   callq loop_header

define void @simple() nounwind {
entry:
  br label %loop

loop:
  call void @loop_header()
  %t0 = tail call i32 @get()
  %t1 = icmp slt i32 %t0, 0
  br i1 %t1, label %done, label %bb

bb:
  call void @loop_latch()
  br label %loop

done:
  call void @exit()
  ret void
}

; CodeGen should move block_a to the top of the loop so that it
; falls through into the loop, avoiding a branch within the loop.

; CHECK-LABEL: slightly_more_involved:
;      CHECK:   jmp .LBB1_1
; CHECK-NEXT:   align
; CHECK-NEXT: .LBB1_4:
; CHECK-NEXT:   callq bar99
; CHECK-NEXT: .LBB1_1:
; CHECK-NEXT:   callq body

define void @slightly_more_involved() nounwind {
entry:
  br label %loop

loop:
  call void @body()
  %t0 = call i32 @get()
  %t1 = icmp slt i32 %t0, 2
  br i1 %t1, label %block_a, label %bb

bb:
  %t2 = call i32 @get()
  %t3 = icmp slt i32 %t2, 99
  br i1 %t3, label %exit, label %loop

block_a:
  call void @bar99()
  br label %loop

exit:
  call void @exit()
  ret void
}

; Same as slightly_more_involved, but block_a is now a CFG diamond with
; fallthrough edges which should be preserved.
; "callq block_a_merge_func" is tail duped.

; CHECK-LABEL: yet_more_involved:
;      CHECK:   jmp .LBB2_1
; CHECK-NEXT:   align
; CHECK-NEXT: .LBB2_5:
; CHECK-NEXT:   callq block_a_true_func
; CHECK-NEXT:   callq block_a_merge_func
; CHECK-NEXT: .LBB2_1:
; CHECK-NEXT:   callq body
;
; LBB2_4
;      CHECK:   callq bar99
; CHECK-NEXT:   callq get
; CHECK-NEXT:   cmpl $2999, %eax
; CHECK-NEXT:   jle .LBB2_5
; CHECK-NEXT:   callq block_a_false_func
; CHECK-NEXT:   callq block_a_merge_func
; CHECK-NEXT:   jmp .LBB2_1

define void @yet_more_involved() nounwind {
entry:
  br label %loop

loop:
  call void @body()
  %t0 = call i32 @get()
  %t1 = icmp slt i32 %t0, 2
  br i1 %t1, label %block_a, label %bb

bb:
  %t2 = call i32 @get()
  %t3 = icmp slt i32 %t2, 99
  br i1 %t3, label %exit, label %loop

block_a:
  call void @bar99()
  %z0 = call i32 @get()
  %z1 = icmp slt i32 %z0, 3000
  br i1 %z1, label %block_a_true, label %block_a_false

block_a_true:
  call void @block_a_true_func()
  br label %block_a_merge

block_a_false:
  call void @block_a_false_func()
  br label %block_a_merge

block_a_merge:
  call void @block_a_merge_func()
  br label %loop

exit:
  call void @exit()
  ret void
}

; CodeGen should move the CFG islands that are part of the loop but don't
; conveniently fit anywhere so that they are at least contiguous with the
; loop.

; CHECK-LABEL: cfg_islands:
;      CHECK:   jmp     .LBB3_1
; CHECK-NEXT:   align
; CHECK-NEXT: .LBB3_7:
; CHECK-NEXT:   callq   bar100
; CHECK-NEXT: .LBB3_1:
; CHECK-NEXT:   callq   loop_header
;      CHECK:   jl .LBB3_7
;      CHECK:   jge .LBB3_3
; CHECK-NEXT:   callq   bar101
; CHECK-NEXT:   jmp     .LBB3_1
; CHECK-NEXT:   align
; CHECK-NEXT: .LBB3_3:
;      CHECK:   jge .LBB3_4
; CHECK-NEXT:   callq   bar102
; CHECK-NEXT:   jmp     .LBB3_1
; CHECK-NEXT: .LBB3_4:
;      CHECK:   jl .LBB3_6
; CHECK-NEXT:   callq   loop_latch
; CHECK-NEXT:   jmp     .LBB3_1
; CHECK-NEXT: .LBB3_6:

define void @cfg_islands() nounwind {
entry:
  br label %loop

loop:
  call void @loop_header()
  %t0 = call i32 @get()
  %t1 = icmp slt i32 %t0, 100
  br i1 %t1, label %block100, label %bb

bb:
  %t2 = call i32 @get()
  %t3 = icmp slt i32 %t2, 101
  br i1 %t3, label %block101, label %bb1

bb1:
  %t4 = call i32 @get()
  %t5 = icmp slt i32 %t4, 102
  br i1 %t5, label %block102, label %bb2

bb2:
  %t6 = call i32 @get()
  %t7 = icmp slt i32 %t6, 103
  br i1 %t7, label %exit, label %bb3

bb3:
  call void @loop_latch()
  br label %loop

exit:
  call void @exit()
  ret void

block100:
  call void @bar100()
  br label %loop

block101:
  call void @bar101()
  br label %loop

block102:
  call void @bar102()
  br label %loop
}

; CHECK-LABEL: check_minsize:
;      CHECK:   jmp   .LBB4_1
; CHECK-NOT:   align
; CHECK-NEXT: .LBB4_2:
; CHECK-NEXT:   callq loop_latch
; CHECK-NEXT: .LBB4_1:
; CHECK-NEXT:   callq loop_header


define void @check_minsize() minsize nounwind {
entry:
  br label %loop

loop:
  call void @loop_header()
  %t0 = tail call i32 @get()
  %t1 = icmp slt i32 %t0, 0
  br i1 %t1, label %done, label %bb

bb:
  call void @loop_latch()
  br label %loop

done:
  call void @exit()
  ret void
}

; This is exactly the same function as slightly_more_involved.
; The difference is that when optimising for size, we do not want
; to see this reordering.

; CHECK-LABEL: slightly_more_involved_2:
; CHECK-NOT:      jmp .LBB5_1
; CHECK:          .LBB5_1:
; CHECK-NEXT:     callq body

define void @slightly_more_involved_2() #0 {
entry:
  br label %loop

loop:
  call void @body()
  %t0 = call i32 @get()
  %t1 = icmp slt i32 %t0, 2
  br i1 %t1, label %block_a, label %bb

bb:
  %t2 = call i32 @get()
  %t3 = icmp slt i32 %t2, 99
  br i1 %t3, label %exit, label %loop

block_a:
  call void @bar99()
  br label %loop

exit:
  call void @exit()
  ret void
}

attributes #0 = { minsize norecurse nounwind optsize readnone uwtable }

declare void @bar99() nounwind
declare void @bar100() nounwind
declare void @bar101() nounwind
declare void @bar102() nounwind
declare void @body() nounwind
declare void @exit() nounwind
declare void @loop_header() nounwind
declare void @loop_latch() nounwind
declare i32 @get() nounwind
declare void @block_a_true_func() nounwind
declare void @block_a_false_func() nounwind
declare void @block_a_merge_func() nounwind
