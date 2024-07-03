; RUN: llc -mtriple=xtensa -disable-block-placement -verify-machineinstrs < %s \
; RUN:   | FileCheck %s

define signext i32 @f_eq(i32 signext %a, ptr %b) nounwind {
; CHECK-LABEL: f_eq:
; CHECK:         l32i a8, a3, 0
; CHECK-NEXT:    beq a2, a8, .LBB0_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    or a2, a8, a8
; CHECK-NEXT:  .LBB0_2:
; CHECK-NEXT:    ret
  %val1 = load i32, ptr %b
  %tst1 = icmp eq i32 %a, %val1
  %val2 = select i1 %tst1, i32 %a, i32 %val1
  ret i32 %val2
}

define signext i32 @f_ne(i32 signext %a, ptr %b) nounwind {
; CHECK-LABEL: f_ne:
; CHECK:         l32i a8, a3, 0
; CHECK-NEXT:    bne a2, a8, .LBB1_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    or a2, a8, a8
; CHECK-NEXT:  .LBB1_2:
; CHECK-NEXT:    ret
  %val1 = load i32, ptr %b
  %tst1 = icmp ne i32 %a, %val1
  %val2 = select i1 %tst1, i32 %a, i32 %val1
  ret i32 %val2
}

define signext i32 @f_ugt(i32 signext %a, ptr %b) nounwind {
; CHECK-LABEL: f_ugt:
; CHECK:         l32i a8, a3, 0
; CHECK-NEXT:    bltu a8, a2, .LBB2_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    or a2, a8, a8
; CHECK-NEXT:  .LBB2_2:
; CHECK-NEXT:    ret
  %val1 = load i32, ptr %b
  %tst1 = icmp ugt i32 %a, %val1
  %val2 = select i1 %tst1, i32 %a, i32 %val1
  ret i32 %val2
}

define signext i32 @f_uge(i32 signext %a, ptr %b) nounwind {
; CHECK-LABEL: f_uge:
; CHECK:         l32i a8, a3, 0
; CHECK-NEXT:    bgeu a2, a8, .LBB3_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    or a2, a8, a8
; CHECK-NEXT:  .LBB3_2:
; CHECK-NEXT:    ret
  %val1 = load i32, ptr %b
  %tst1 = icmp uge i32 %a, %val1
  %val2 = select i1 %tst1, i32 %a, i32 %val1
  ret i32 %val2
}

define signext i32 @f_ult(i32 signext %a, ptr %b) nounwind {
; CHECK-LABEL: f_ult:
; CHECK:         l32i a8, a3, 0
; CHECK-NEXT:    bltu a2, a8, .LBB4_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    or a2, a8, a8
; CHECK-NEXT:  .LBB4_2:
; CHECK-NEXT:    ret
  %val1 = load i32, ptr %b
  %tst1 = icmp ult i32 %a, %val1
  %val2 = select i1 %tst1, i32 %a, i32 %val1
  ret i32 %val2
}

define signext i32 @f_ule(i32 signext %a, ptr %b) nounwind {
; CHECK-LABEL: f_ule:
; CHECK:         l32i a8, a3, 0
; CHECK-NEXT:    bgeu a8, a2, .LBB5_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    or a2, a8, a8
; CHECK-NEXT:  .LBB5_2:
; CHECK-NEXT:    ret
  %val1 = load i32, ptr %b
  %tst1 = icmp ule i32 %a, %val1
  %val2 = select i1 %tst1, i32 %a, i32 %val1
  ret i32 %val2
}

define signext i32 @f_sgt(i32 signext %a, ptr %b) nounwind {
; CHECK-LABEL: f_sgt:
; CHECK:         l32i a8, a3, 0
; CHECK-NEXT:    blt a8, a2, .LBB6_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    or a2, a8, a8
; CHECK-NEXT:  .LBB6_2:
; CHECK-NEXT:    ret
  %val1 = load i32, ptr %b
  %tst1 = icmp sgt i32 %a, %val1
  %val2 = select i1 %tst1, i32 %a, i32 %val1
  ret i32 %val2
}

define signext i32 @f_sge(i32 signext %a, ptr %b) nounwind {
; CHECK-LABEL: f_sge:
; CHECK:         l32i a8, a3, 0
; CHECK-NEXT:    bge a2, a8, .LBB7_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    or a2, a8, a8
; CHECK-NEXT:  .LBB7_2:
; CHECK-NEXT:    ret
  %val1 = load i32, ptr %b
  %tst1 = icmp sge i32 %a, %val1
  %val2 = select i1 %tst1, i32 %a, i32 %val1
  ret i32 %val2
}

define signext i32 @f_slt(i32 signext %a, ptr %b) nounwind {
; CHECK-LABEL: f_slt:
; CHECK:         l32i a8, a3, 0
; CHECK-NEXT:    blt a2, a8, .LBB8_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    or a2, a8, a8
; CHECK-NEXT:  .LBB8_2:
; CHECK-NEXT:    ret
  %val1 = load i32, ptr %b
  %tst1 = icmp slt i32 %a, %val1
  %val2 = select i1 %tst1, i32 %a, i32 %val1
  ret i32 %val2
}

define signext i32 @f_sle(i32 signext %a, ptr %b) nounwind {
; CHECK-LABEL: f_sle:
; CHECK:         l32i a8, a3, 0
; CHECK-NEXT:    bge a8, a2, .LBB9_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    or a2, a8, a8
; CHECK-NEXT:  .LBB9_2:
; CHECK-NEXT:    ret
  %val1 = load i32, ptr %b
  %tst1 = icmp sle i32 %a, %val1
  %val2 = select i1 %tst1, i32 %a, i32 %val1
  ret i32 %val2
}

define signext i32 @f_slt_imm(i32 signext %a, ptr %b) nounwind {
; CHECK-LABEL: f_slt_imm:
; CHECK:         movi a8, 1
; CHECK-NEXT:    blt a2, a8, .LBB10_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    l32i a2, a3, 0
; CHECK-NEXT:  .LBB10_2:
; CHECK-NEXT:    ret
  %val1 = load i32, ptr %b
  %tst1 = icmp slt i32 %a, 1
  %val2 = select i1 %tst1, i32 %a, i32 %val1
  ret i32 %val2
}

define signext i32 @f_sgt_imm(i32 signext %a, ptr %b) nounwind {
; CHECK-LABEL: f_sgt_imm:
; CHECK:         movi a8, -1
; CHECK-NEXT:    blt a8, a2, .LBB11_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    l32i a2, a3, 0
; CHECK-NEXT:  .LBB11_2:
; CHECK-NEXT:    ret
  %val1 = load i32, ptr %b
  %tst1 = icmp sgt i32 %a, -1
  %val2 = select i1 %tst1, i32 %a, i32 %val1
  ret i32 %val2
}

define signext i32 @f_ult_imm(i32 signext %a, ptr %b) nounwind {
; CHECK-LABEL: f_ult_imm:
; CHECK:         movi a8, 1024
; CHECK-NEXT:    bltu a2, a8, .LBB12_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    l32i a2, a3, 0
; CHECK-NEXT:  .LBB12_2:
; CHECK-NEXT:    ret
  %val1 = load i32, ptr %b
  %tst1 = icmp ult i32 %a, 1024
  %val2 = select i1 %tst1, i32 %a, i32 %val1
  ret i32 %val2
}
