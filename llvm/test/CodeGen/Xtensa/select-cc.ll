; RUN: llc -mtriple=xtensa -disable-block-placement -verify-machineinstrs < %s \
; RUN:   | FileCheck %s

define i32 @f_eq(i32 %a, ptr %b) nounwind {
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

define i32 @f_ne(i32 %a, ptr %b) nounwind {
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

define i32 @f_ugt(i32 %a, ptr %b) nounwind {
; CHECK-LABEL: f_ugt:
; CHECK:         or a8, a2, a2
; CHECK-NEXT:    l32i a2, a3, 0
; CHECK-NEXT:    bgeu a2, a8, .LBB2_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    or a2, a8, a8
; CHECK-NEXT:  .LBB2_2:
; CHECK-NEXT:    ret
  %val1 = load i32, ptr %b
  %tst1 = icmp ugt i32 %a, %val1
  %val2 = select i1 %tst1, i32 %a, i32 %val1
  ret i32 %val2
}

define i32 @f_uge(i32 %a, ptr %b) nounwind {
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

define i32 @f_ult(i32 %a, ptr %b) nounwind {
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

define i32 @f_ule(i32 %a, ptr %b) nounwind {
; CHECK-LABEL: f_ule:
; CHECK:         or a8, a2, a2
; CHECK-NEXT:    l32i a2, a3, 0
; CHECK-NEXT:    bltu a2, a8, .LBB5_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    or a2, a8, a8
; CHECK-NEXT:  .LBB5_2:
; CHECK-NEXT:    ret
  %val1 = load i32, ptr %b
  %tst1 = icmp ule i32 %a, %val1
  %val2 = select i1 %tst1, i32 %a, i32 %val1
  ret i32 %val2
}

define i32 @f_sgt(i32 %a, ptr %b) nounwind {
; CHECK-LABEL: f_sgt:
; CHECK:         or a8, a2, a2
; CHECK-NEXT:    l32i a2, a3, 0
; CHECK-NEXT:    bge a2, a8, .LBB6_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    or a2, a8, a8
; CHECK-NEXT:  .LBB6_2:
; CHECK-NEXT:    ret
  %val1 = load i32, ptr %b
  %tst1 = icmp sgt i32 %a, %val1
  %val2 = select i1 %tst1, i32 %a, i32 %val1
  ret i32 %val2
}

define i32 @f_sge(i32 %a, ptr %b) nounwind {
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

define i32 @f_slt(i32 %a, ptr %b) nounwind {
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

define i32 @f_sle(i32 %a, ptr %b) nounwind {
; CHECK-LABEL: f_sle:
; CHECK:         or a8, a2, a2
; CHECK-NEXT:    l32i a2, a3, 0
; CHECK-NEXT:    blt a2, a8, .LBB9_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    or a2, a8, a8
; CHECK-NEXT:  .LBB9_2:
; CHECK-NEXT:    ret
  %val1 = load i32, ptr %b
  %tst1 = icmp sle i32 %a, %val1
  %val2 = select i1 %tst1, i32 %a, i32 %val1
  ret i32 %val2
}

define i32 @f_slt_imm(i32 %a, ptr %b) nounwind {
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

define i32 @f_sgt_imm(i32 %a, ptr %b) nounwind {
; CHECK-LABEL: f_sgt_imm:
; CHECK:         or a8, a2, a2
; CHECK-NEXT:    l32i a2, a3, 0
; CHECK-NEXT:    movi a9, -1
; CHECK-NEXT:    bge a9, a8, .LBB11_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    or a2, a8, a8
; CHECK-NEXT:  .LBB11_2:
; CHECK-NEXT:    ret
  %val1 = load i32, ptr %b
  %tst1 = icmp sgt i32 %a, -1
  %val2 = select i1 %tst1, i32 %a, i32 %val1
  ret i32 %val2
}

define i32 @f_ult_imm(i32 %a, ptr %b) nounwind {
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

; Tests for i64 operands

define i64 @f_eq_i64(i64 %a, ptr %b) nounwind {
; CHECK-LABEL: f_eq_i64:
; CHECK:         l32i a8, a4, 4
; CHECK-NEXT:    xor a9, a3, a8
; CHECK-NEXT:    l32i a11, a4, 0
; CHECK-NEXT:    xor a10, a2, a11
; CHECK-NEXT:    or a9, a10, a9
; CHECK-NEXT:    movi a10, 0
; CHECK-NEXT:    beq a9, a10, .LBB13_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    or a2, a11, a11
; CHECK-NEXT:  .LBB13_2:
; CHECK-NEXT:    beq a9, a10, .LBB13_4
; CHECK-NEXT:  # %bb.3:
; CHECK-NEXT:    or a3, a8, a8
; CHECK-NEXT:  .LBB13_4:
; CHECK-NEXT:    ret
  %val1 = load i64, ptr %b
  %tst1 = icmp eq i64 %a, %val1
  %val2 = select i1 %tst1, i64 %a, i64 %val1
  ret i64 %val2
}

define i64 @f_ne_i64(i64 %a, ptr %b) nounwind {
; CHECK-LABEL: f_ne_i64:
; CHECK:         l32i a8, a4, 4
; CHECK-NEXT:    xor a9, a3, a8
; CHECK-NEXT:    l32i a11, a4, 0
; CHECK-NEXT:    xor a10, a2, a11
; CHECK-NEXT:    or a9, a10, a9
; CHECK-NEXT:    movi a10, 0
; CHECK-NEXT:    bne a9, a10, .LBB14_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    or a2, a11, a11
; CHECK-NEXT:  .LBB14_2:
; CHECK-NEXT:    bne a9, a10, .LBB14_4
; CHECK-NEXT:  # %bb.3:
; CHECK-NEXT:    or a3, a8, a8
; CHECK-NEXT:  .LBB14_4:
; CHECK-NEXT:    ret
  %val1 = load i64, ptr %b
  %tst1 = icmp ne i64 %a, %val1
  %val2 = select i1 %tst1, i64 %a, i64 %val1
  ret i64 %val2
}

define i64 @f_ugt_i64(i64 %a, ptr %b) nounwind {
; CHECK-LABEL: f_ugt_i64:
; CHECK:         l32i a8, a4, 4
; CHECK-NEXT:    movi a9, 0
; CHECK-NEXT:    movi a10, 1
; CHECK-NEXT:    or a7, a10, a10
; CHECK-NEXT:    bltu a8, a3, .LBB15_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    or a7, a9, a9
; CHECK-NEXT:  .LBB15_2:
; CHECK-NEXT:    l32i a11, a4, 0
; CHECK-NEXT:    bltu a11, a2, .LBB15_4
; CHECK-NEXT:  # %bb.3:
; CHECK-NEXT:    or a10, a9, a9
; CHECK-NEXT:  .LBB15_4:
; CHECK-NEXT:    beq a3, a8, .LBB15_6
; CHECK-NEXT:  # %bb.5:
; CHECK-NEXT:    or a10, a7, a7
; CHECK-NEXT:  .LBB15_6:
; CHECK-NEXT:    bne a10, a9, .LBB15_8
; CHECK-NEXT:  # %bb.7:
; CHECK-NEXT:    or a2, a11, a11
; CHECK-NEXT:  .LBB15_8:
; CHECK-NEXT:    bne a10, a9, .LBB15_10
; CHECK-NEXT:  # %bb.9:
; CHECK-NEXT:    or a3, a8, a8
; CHECK-NEXT:  .LBB15_10:
; CHECK-NEXT:    ret
  %val1 = load i64, ptr %b
  %tst1 = icmp ugt i64 %a, %val1
  %val2 = select i1 %tst1, i64 %a, i64 %val1
  ret i64 %val2
}

define i64 @f_uge_i64(i64 %a, ptr %b) nounwind {
; CHECK-LABEL: f_uge_i64:
; CHECK:         l32i a8, a4, 4
; CHECK-NEXT:    movi a9, 0
; CHECK-NEXT:    movi a10, 1
; CHECK-NEXT:    or a7, a10, a10
; CHECK-NEXT:    bgeu a3, a8, .LBB16_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    or a7, a9, a9
; CHECK-NEXT:  .LBB16_2:
; CHECK-NEXT:    l32i a11, a4, 0
; CHECK-NEXT:    bgeu a2, a11, .LBB16_4
; CHECK-NEXT:  # %bb.3:
; CHECK-NEXT:    or a10, a9, a9
; CHECK-NEXT:  .LBB16_4:
; CHECK-NEXT:    beq a3, a8, .LBB16_6
; CHECK-NEXT:  # %bb.5:
; CHECK-NEXT:    or a10, a7, a7
; CHECK-NEXT:  .LBB16_6:
; CHECK-NEXT:    bne a10, a9, .LBB16_8
; CHECK-NEXT:  # %bb.7:
; CHECK-NEXT:    or a2, a11, a11
; CHECK-NEXT:  .LBB16_8:
; CHECK-NEXT:    bne a10, a9, .LBB16_10
; CHECK-NEXT:  # %bb.9:
; CHECK-NEXT:    or a3, a8, a8
; CHECK-NEXT:  .LBB16_10:
; CHECK-NEXT:    ret
  %val1 = load i64, ptr %b
  %tst1 = icmp uge i64 %a, %val1
  %val2 = select i1 %tst1, i64 %a, i64 %val1
  ret i64 %val2
}

define i64 @f_ult_i64(i64 %a, ptr %b) nounwind {
; CHECK-LABEL: f_ult_i64:
; CHECK:         l32i a8, a4, 4
; CHECK-NEXT:    movi a9, 0
; CHECK-NEXT:    movi a10, 1
; CHECK-NEXT:    or a7, a10, a10
; CHECK-NEXT:    bltu a3, a8, .LBB17_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    or a7, a9, a9
; CHECK-NEXT:  .LBB17_2:
; CHECK-NEXT:    l32i a11, a4, 0
; CHECK-NEXT:    bltu a2, a11, .LBB17_4
; CHECK-NEXT:  # %bb.3:
; CHECK-NEXT:    or a10, a9, a9
; CHECK-NEXT:  .LBB17_4:
; CHECK-NEXT:    beq a3, a8, .LBB17_6
; CHECK-NEXT:  # %bb.5:
; CHECK-NEXT:    or a10, a7, a7
; CHECK-NEXT:  .LBB17_6:
; CHECK-NEXT:    bne a10, a9, .LBB17_8
; CHECK-NEXT:  # %bb.7:
; CHECK-NEXT:    or a2, a11, a11
; CHECK-NEXT:  .LBB17_8:
; CHECK-NEXT:    bne a10, a9, .LBB17_10
; CHECK-NEXT:  # %bb.9:
; CHECK-NEXT:    or a3, a8, a8
; CHECK-NEXT:  .LBB17_10:
; CHECK-NEXT:    ret
  %val1 = load i64, ptr %b
  %tst1 = icmp ult i64 %a, %val1
  %val2 = select i1 %tst1, i64 %a, i64 %val1
  ret i64 %val2
}

define i64 @f_ule_i64(i64 %a, ptr %b) nounwind {
; CHECK-LABEL: f_ule_i64:
; CHECK:         l32i a8, a4, 4
; CHECK-NEXT:    movi a9, 0
; CHECK-NEXT:    movi a10, 1
; CHECK-NEXT:    or a7, a10, a10
; CHECK-NEXT:    bgeu a8, a3, .LBB18_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    or a7, a9, a9
; CHECK-NEXT:  .LBB18_2:
; CHECK-NEXT:    l32i a11, a4, 0
; CHECK-NEXT:    bgeu a11, a2, .LBB18_4
; CHECK-NEXT:  # %bb.3:
; CHECK-NEXT:    or a10, a9, a9
; CHECK-NEXT:  .LBB18_4:
; CHECK-NEXT:    beq a3, a8, .LBB18_6
; CHECK-NEXT:  # %bb.5:
; CHECK-NEXT:    or a10, a7, a7
; CHECK-NEXT:  .LBB18_6:
; CHECK-NEXT:    bne a10, a9, .LBB18_8
; CHECK-NEXT:  # %bb.7:
; CHECK-NEXT:    or a2, a11, a11
; CHECK-NEXT:  .LBB18_8:
; CHECK-NEXT:    bne a10, a9, .LBB18_10
; CHECK-NEXT:  # %bb.9:
; CHECK-NEXT:    or a3, a8, a8
; CHECK-NEXT:  .LBB18_10:
; CHECK-NEXT:    ret
  %val1 = load i64, ptr %b
  %tst1 = icmp ule i64 %a, %val1
  %val2 = select i1 %tst1, i64 %a, i64 %val1
  ret i64 %val2
}

define i64 @f_sgt_i64(i64 %a, ptr %b) nounwind {
; CHECK-LABEL: f_sgt_i64:
; CHECK:         l32i a8, a4, 4
; CHECK-NEXT:    movi a9, 0
; CHECK-NEXT:    movi a10, 1
; CHECK-NEXT:    or a7, a10, a10
; CHECK-NEXT:    blt a8, a3, .LBB19_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    or a7, a9, a9
; CHECK-NEXT:  .LBB19_2:
; CHECK-NEXT:    l32i a11, a4, 0
; CHECK-NEXT:    bltu a11, a2, .LBB19_4
; CHECK-NEXT:  # %bb.3:
; CHECK-NEXT:    or a10, a9, a9
; CHECK-NEXT:  .LBB19_4:
; CHECK-NEXT:    beq a3, a8, .LBB19_6
; CHECK-NEXT:  # %bb.5:
; CHECK-NEXT:    or a10, a7, a7
; CHECK-NEXT:  .LBB19_6:
; CHECK-NEXT:    bne a10, a9, .LBB19_8
; CHECK-NEXT:  # %bb.7:
; CHECK-NEXT:    or a2, a11, a11
; CHECK-NEXT:  .LBB19_8:
; CHECK-NEXT:    bne a10, a9, .LBB19_10
; CHECK-NEXT:  # %bb.9:
; CHECK-NEXT:    or a3, a8, a8
; CHECK-NEXT:  .LBB19_10:
; CHECK-NEXT:    ret
  %val1 = load i64, ptr %b
  %tst1 = icmp sgt i64 %a, %val1
  %val2 = select i1 %tst1, i64 %a, i64 %val1
  ret i64 %val2
}

define i64 @f_sge_i64(i64 %a, ptr %b) nounwind {
; CHECK-LABEL: f_sge_i64:
; CHECK:         l32i a8, a4, 4
; CHECK-NEXT:    movi a9, 0
; CHECK-NEXT:    movi a10, 1
; CHECK-NEXT:    or a7, a10, a10
; CHECK-NEXT:    bge a3, a8, .LBB20_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    or a7, a9, a9
; CHECK-NEXT:  .LBB20_2:
; CHECK-NEXT:    l32i a11, a4, 0
; CHECK-NEXT:    bgeu a2, a11, .LBB20_4
; CHECK-NEXT:  # %bb.3:
; CHECK-NEXT:    or a10, a9, a9
; CHECK-NEXT:  .LBB20_4:
; CHECK-NEXT:    beq a3, a8, .LBB20_6
; CHECK-NEXT:  # %bb.5:
; CHECK-NEXT:    or a10, a7, a7
; CHECK-NEXT:  .LBB20_6:
; CHECK-NEXT:    bne a10, a9, .LBB20_8
; CHECK-NEXT:  # %bb.7:
; CHECK-NEXT:    or a2, a11, a11
; CHECK-NEXT:  .LBB20_8:
; CHECK-NEXT:    bne a10, a9, .LBB20_10
; CHECK-NEXT:  # %bb.9:
; CHECK-NEXT:    or a3, a8, a8
; CHECK-NEXT:  .LBB20_10:
; CHECK-NEXT:    ret
  %val1 = load i64, ptr %b
  %tst1 = icmp sge i64 %a, %val1
  %val2 = select i1 %tst1, i64 %a, i64 %val1
  ret i64 %val2
}

define i64 @f_slt_i64(i64 %a, ptr %b) nounwind {
; CHECK-LABEL: f_slt_i64:
; CHECK:         l32i a8, a4, 4
; CHECK-NEXT:    movi a9, 0
; CHECK-NEXT:    movi a10, 1
; CHECK-NEXT:    or a7, a10, a10
; CHECK-NEXT:    blt a3, a8, .LBB21_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    or a7, a9, a9
; CHECK-NEXT:  .LBB21_2:
; CHECK-NEXT:    l32i a11, a4, 0
; CHECK-NEXT:    bltu a2, a11, .LBB21_4
; CHECK-NEXT:  # %bb.3:
; CHECK-NEXT:    or a10, a9, a9
; CHECK-NEXT:  .LBB21_4:
; CHECK-NEXT:    beq a3, a8, .LBB21_6
; CHECK-NEXT:  # %bb.5:
; CHECK-NEXT:    or a10, a7, a7
; CHECK-NEXT:  .LBB21_6:
; CHECK-NEXT:    bne a10, a9, .LBB21_8
; CHECK-NEXT:  # %bb.7:
; CHECK-NEXT:    or a2, a11, a11
; CHECK-NEXT:  .LBB21_8:
; CHECK-NEXT:    bne a10, a9, .LBB21_10
; CHECK-NEXT:  # %bb.9:
; CHECK-NEXT:    or a3, a8, a8
; CHECK-NEXT:  .LBB21_10:
; CHECK-NEXT:    ret
  %val1 = load i64, ptr %b
  %tst1 = icmp slt i64 %a, %val1
  %val2 = select i1 %tst1, i64 %a, i64 %val1
  ret i64 %val2
}

define i64 @f_sle_i64(i64 %a, ptr %b) nounwind {
; CHECK-LABEL: f_sle_i64:
; CHECK:         l32i a8, a4, 4
; CHECK-NEXT:    movi a9, 0
; CHECK-NEXT:    movi a10, 1
; CHECK-NEXT:    or a7, a10, a10
; CHECK-NEXT:    bge a8, a3, .LBB22_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    or a7, a9, a9
; CHECK-NEXT:  .LBB22_2:
; CHECK-NEXT:    l32i a11, a4, 0
; CHECK-NEXT:    bgeu a11, a2, .LBB22_4
; CHECK-NEXT:  # %bb.3:
; CHECK-NEXT:    or a10, a9, a9
; CHECK-NEXT:  .LBB22_4:
; CHECK-NEXT:    beq a3, a8, .LBB22_6
; CHECK-NEXT:  # %bb.5:
; CHECK-NEXT:    or a10, a7, a7
; CHECK-NEXT:  .LBB22_6:
; CHECK-NEXT:    bne a10, a9, .LBB22_8
; CHECK-NEXT:  # %bb.7:
; CHECK-NEXT:    or a2, a11, a11
; CHECK-NEXT:  .LBB22_8:
; CHECK-NEXT:    bne a10, a9, .LBB22_10
; CHECK-NEXT:  # %bb.9:
; CHECK-NEXT:    or a3, a8, a8
; CHECK-NEXT:  .LBB22_10:
; CHECK-NEXT:    ret
  %val1 = load i64, ptr %b
  %tst1 = icmp sle i64 %a, %val1
  %val2 = select i1 %tst1, i64 %a, i64 %val1
  ret i64 %val2
}
