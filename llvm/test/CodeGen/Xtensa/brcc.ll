; RUN: llc -mtriple=xtensa -disable-block-placement -verify-machineinstrs < %s \
; RUN:   | FileCheck %s

define i32 @brcc_sgt(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: brcc_sgt:
; CHECK:         bge a3, a2, .LBB0_2
; CHECK-NEXT:    j .LBB0_1
; CHECK-NEXT:  .LBB0_1: # %t1
; CHECK-NEXT:    addi a2, a2, 4
; CHECK-NEXT:    j .LBB0_3
; CHECK-NEXT:  .LBB0_2: # %t2
; CHECK-NEXT:    addi a2, a3, 8
; CHECK-NEXT:  .LBB0_3: # %exit
; CHECK-NEXT:    ret
  %wb = icmp sgt i32 %a, %b
  br i1 %wb, label %t1, label %t2
t1:
  %t1v = add i32 %a, 4
  br label %exit
t2:
  %t2v = add i32 %b, 8
  br label %exit
exit:
  %v = phi i32 [ %t1v, %t1 ], [ %t2v, %t2 ]
  ret i32 %v
}

define i32 @brcc_ugt(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: brcc_ugt:
; CHECK:         bgeu a3, a2, .LBB1_2
; CHECK-NEXT:    j .LBB1_1
; CHECK-NEXT:  .LBB1_1: # %t1
; CHECK-NEXT:    addi a2, a2, 4
; CHECK-NEXT:    j .LBB1_3
; CHECK-NEXT:  .LBB1_2: # %t2
; CHECK-NEXT:    addi a2, a3, 8
; CHECK-NEXT:  .LBB1_3: # %exit
; CHECK-NEXT:    ret
  %wb = icmp ugt i32 %a, %b
  br i1 %wb, label %t1, label %t2
t1:
  %t1v = add i32 %a, 4
  br label %exit
t2:
  %t2v = add i32 %b, 8
  br label %exit
exit:
  %v = phi i32 [ %t1v, %t1 ], [ %t2v, %t2 ]
  ret i32 %v
}

define i32 @brcc_sle(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: brcc_sle:
; CHECK:         blt a3, a2, .LBB2_2
; CHECK-NEXT:    j .LBB2_1
; CHECK-NEXT:  .LBB2_1: # %t1
; CHECK-NEXT:    addi a2, a2, 4
; CHECK-NEXT:    j .LBB2_3
; CHECK-NEXT:  .LBB2_2: # %t2
; CHECK-NEXT:    addi a2, a3, 8
; CHECK-NEXT:  .LBB2_3: # %exit
; CHECK-NEXT:    ret
  %wb = icmp sle i32 %a, %b
  br i1 %wb, label %t1, label %t2
t1:
  %t1v = add i32 %a, 4
  br label %exit
t2:
  %t2v = add i32 %b, 8
  br label %exit
exit:
  %v = phi i32 [ %t1v, %t1 ], [ %t2v, %t2 ]
  ret i32 %v
}

define i32 @brcc_ule(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: brcc_ule:
; CHECK:         bltu a3, a2, .LBB3_2
; CHECK-NEXT:    j .LBB3_1
; CHECK-NEXT:  .LBB3_1: # %t1
; CHECK-NEXT:    addi a2, a2, 4
; CHECK-NEXT:    j .LBB3_3
; CHECK-NEXT:  .LBB3_2: # %t2
; CHECK-NEXT:    addi a2, a3, 8
; CHECK-NEXT:  .LBB3_3: # %exit
; CHECK-NEXT:    ret
  %wb = icmp ule i32 %a, %b
  br i1 %wb, label %t1, label %t2
t1:
  %t1v = add i32 %a, 4
  br label %exit
t2:
  %t2v = add i32 %b, 8
  br label %exit
exit:
  %v = phi i32 [ %t1v, %t1 ], [ %t2v, %t2 ]
  ret i32 %v
}

define i32 @brcc_eq(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: brcc_eq:
; CHECK:         bne a2, a3, .LBB4_2
; CHECK-NEXT:    j .LBB4_1
; CHECK-NEXT:  .LBB4_1: # %t1
; CHECK-NEXT:    addi a2, a2, 4
; CHECK-NEXT:    j .LBB4_3
; CHECK-NEXT:  .LBB4_2: # %t2
; CHECK-NEXT:    addi a2, a3, 8
; CHECK-NEXT:  .LBB4_3: # %exit
; CHECK-NEXT:    ret
  %wb = icmp eq i32 %a, %b
  br i1 %wb, label %t1, label %t2
t1:
  %t1v = add i32 %a, 4
  br label %exit
t2:
  %t2v = add i32 %b, 8
  br label %exit
exit:
  %v = phi i32 [ %t1v, %t1 ], [ %t2v, %t2 ]
  ret i32 %v
}

define i32 @brcc_ne(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: brcc_ne:
; CHECK:         beq a2, a3, .LBB5_2
; CHECK-NEXT:    j .LBB5_1
; CHECK-NEXT:  .LBB5_1: # %t1
; CHECK-NEXT:    addi a2, a2, 4
; CHECK-NEXT:    j .LBB5_3
; CHECK-NEXT:  .LBB5_2: # %t2
; CHECK-NEXT:    addi a2, a3, 8
; CHECK-NEXT:  .LBB5_3: # %exit
; CHECK-NEXT:    ret
  %wb = icmp ne i32 %a, %b
  br i1 %wb, label %t1, label %t2
t1:
  %t1v = add i32 %a, 4
  br label %exit
t2:
  %t2v = add i32 %b, 8
  br label %exit
exit:
  %v = phi i32 [ %t1v, %t1 ], [ %t2v, %t2 ]
  ret i32 %v
}

define i32 @brcc_ge(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: brcc_ge:
; CHECK:         blt a2, a3, .LBB6_2
; CHECK-NEXT:    j .LBB6_1
; CHECK-NEXT:  .LBB6_1: # %t1
; CHECK-NEXT:    addi a2, a2, 4
; CHECK-NEXT:    j .LBB6_3
; CHECK-NEXT:  .LBB6_2: # %t2
; CHECK-NEXT:    addi a2, a3, 8
; CHECK-NEXT:  .LBB6_3: # %exit
; CHECK-NEXT:    ret
  %wb = icmp sge i32 %a, %b
  br i1 %wb, label %t1, label %t2
t1:
  %t1v = add i32 %a, 4
  br label %exit
t2:
  %t2v = add i32 %b, 8
  br label %exit
exit:
  %v = phi i32 [ %t1v, %t1 ], [ %t2v, %t2 ]
  ret i32 %v
}

define i32 @brcc_lt(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: brcc_lt:
; CHECK:         bge a2, a3, .LBB7_2
; CHECK-NEXT:    j .LBB7_1
; CHECK-NEXT:  .LBB7_1: # %t1
; CHECK-NEXT:    addi a2, a2, 4
; CHECK-NEXT:    j .LBB7_3
; CHECK-NEXT:  .LBB7_2: # %t2
; CHECK-NEXT:    addi a2, a3, 8
; CHECK-NEXT:  .LBB7_3: # %exit
; CHECK-NEXT:    ret
  %wb = icmp slt i32 %a, %b
  br i1 %wb, label %t1, label %t2
t1:
  %t1v = add i32 %a, 4
  br label %exit
t2:
  %t2v = add i32 %b, 8
  br label %exit
exit:
  %v = phi i32 [ %t1v, %t1 ], [ %t2v, %t2 ]
  ret i32 %v
}

define i32 @brcc_uge(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: brcc_uge:
; CHECK:         bltu a2, a3, .LBB8_2
; CHECK-NEXT:    j .LBB8_1
; CHECK-NEXT:  .LBB8_1: # %t1
; CHECK-NEXT:    addi a2, a2, 4
; CHECK-NEXT:    j .LBB8_3
; CHECK-NEXT:  .LBB8_2: # %t2
; CHECK-NEXT:    addi a2, a3, 8
; CHECK-NEXT:  .LBB8_3: # %exit
; CHECK-NEXT:    ret
  %wb = icmp uge i32 %a, %b
  br i1 %wb, label %t1, label %t2
t1:
  %t1v = add i32 %a, 4
  br label %exit
t2:
  %t2v = add i32 %b, 8
  br label %exit
exit:
  %v = phi i32 [ %t1v, %t1 ], [ %t2v, %t2 ]
  ret i32 %v
}

define i32 @brcc_ult(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: brcc_ult:
; CHECK:         bgeu a2, a3, .LBB9_2
; CHECK-NEXT:    j .LBB9_1
; CHECK-NEXT:  .LBB9_1: # %t1
; CHECK-NEXT:    addi a2, a2, 4
; CHECK-NEXT:    j .LBB9_3
; CHECK-NEXT:  .LBB9_2: # %t2
; CHECK-NEXT:    addi a2, a3, 8
; CHECK-NEXT:  .LBB9_3: # %exit
; CHECK-NEXT:    ret
  %wb = icmp ult i32 %a, %b
  br i1 %wb, label %t1, label %t2
t1:
  %t1v = add i32 %a, 4
  br label %exit
t2:
  %t2v = add i32 %b, 8
  br label %exit
exit:
  %v = phi i32 [ %t1v, %t1 ], [ %t2v, %t2 ]
  ret i32 %v
}
