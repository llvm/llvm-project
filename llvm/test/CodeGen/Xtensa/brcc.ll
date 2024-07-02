; RUN: llc -march=xtensa < %s | FileCheck %s

; CHECK-LABEL: brcc1:
; CHECK: bge   a3, a2, .LBB0_2
; CHECK: addi  a2, a2, 4
; CHECK: .LBB0_2:
define i32 @brcc1(i32 %a, i32 %b) nounwind {
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

; CHECK-LABEL: brcc2
; CHECK: bgeu  a3, a2, .LBB1_2
; CHECK: addi  a2, a2, 4
; CHECK: .LBB1_2:
define i32 @brcc2(i32 %a, i32 %b) nounwind {
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

; CHECK-LABEL: brcc3:
; CHECK: blt   a3, a2, .LBB2_2
; CHECK: addi  a2, a2, 4
; CHECK: .LBB2_2:
define i32 @brcc3(i32 %a, i32 %b) nounwind {
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

; CHECK-LABEL: brcc4
; CHECK: bltu  a3, a2, .LBB3_2
; CHECK: addi  a2, a2, 4
; CHECK: .LBB3_2:
define i32 @brcc4(i32 %a, i32 %b) nounwind {
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
