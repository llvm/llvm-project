; RUN: llc -march=xtensa < %s | FileCheck %s

; CHECK-LABEL: brcc_sgt:
; CHECK: bge   a3, a2, .LBB0_2
; CHECK: addi  a2, a2, 4
; CHECK: .LBB0_2:
define i32 @brcc_sgt(i32 %a, i32 %b) nounwind {
entry:
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

; CHECK-LABEL: brcc_ugt
; CHECK: bgeu  a3, a2, .LBB1_2
; CHECK: addi  a2, a2, 4
; CHECK: .LBB1_2:
define i32 @brcc_ugt(i32 %a, i32 %b) nounwind {
entry:
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

; CHECK-LABEL: brcc_sle:
; CHECK: blt   a3, a2, .LBB2_2
; CHECK: addi  a2, a2, 4
; CHECK: .LBB2_2:
define i32 @brcc_sle(i32 %a, i32 %b) nounwind {
entry:
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

; CHECK-LABEL: brcc_ule
; CHECK: bltu  a3, a2, .LBB3_2
; CHECK: addi  a2, a2, 4
; CHECK: .LBB3_2:
define i32 @brcc_ule(i32 %a, i32 %b) nounwind {
entry:
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

; CHECK-LABEL: brcc_eq:
; CHECK: bne   a2, a3, .LBB4_2
; CHECK: addi  a2, a2, 4
; CHECK: .LBB4_2:
define i32 @brcc_eq(i32 %a, i32 %b) nounwind {
entry:
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

; CHECK-LABEL: brcc_ne:
; CHECK: beq   a2, a3, .LBB5_2
; CHECK: addi  a2, a2, 4
; CHECK: .LBB5_2:
define i32 @brcc_ne(i32 %a, i32 %b) nounwind {
entry:
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

; CHECK-LABEL: brcc_ge:
; CHECK: blt   a2, a3, .LBB6_2
; CHECK: addi  a2, a2, 4
; CHECK: .LBB6_2:
define i32 @brcc_ge(i32 %a, i32 %b) nounwind {
entry:
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

; CHECK-LABEL: brcc_lt:
; CHECK: bge   a2, a3, .LBB7_2
; CHECK: addi  a2, a2, 4
; CHECK: .LBB7_2:
define i32 @brcc_lt(i32 %a, i32 %b) nounwind {
entry:
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

; CHECK-LABEL: brcc_uge:
; CHECK: bltu  a2, a3, .LBB8_2
; CHECK: addi  a2, a2, 4
; CHECK: .LBB8_2:
define i32 @brcc_uge(i32 %a, i32 %b) nounwind {
entry:
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

; CHECK-LABEL: brcc_ult:
; CHECK: bgeu  a2, a3, .LBB9_2
; CHECK: addi  a2, a2, 4
; CHECK: .LBB9_2:
define i32 @brcc_ult(i32 %a, i32 %b) nounwind {
entry:
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
