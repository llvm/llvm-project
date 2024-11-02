; RUN: llc < %s -mtriple=xtensa -O0 | FileCheck %s

define i32 @f_eq(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: f_eq:
; CHECK:         addi a8, a1, -16
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    movi a8, 0
; CHECK-NEXT:    s32i a8, a1, 0 # 4-byte Folded Spill
; CHECK-NEXT:    movi a8, 1
; CHECK-NEXT:    s32i a8, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:    beq a2, a3, .LBB0_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    l32i a8, a1, 0 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB0_2:
; CHECK-NEXT:    l32i a2, a1, 4 # 4-byte Folded Reload
; CHECK-NEXT:    addi a8, a1, 16
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    ret

  %cond = icmp eq i32 %a, %b
  %res = zext i1 %cond to i32
  ret i32 %res
}

define i32 @f_slt(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: f_slt:
; CHECK:         addi a8, a1, -16
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    movi a8, 0
; CHECK-NEXT:    s32i a8, a1, 0 # 4-byte Folded Spill
; CHECK-NEXT:    movi a8, 1
; CHECK-NEXT:    s32i a8, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:    blt a2, a3, .LBB1_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    l32i a8, a1, 0 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB1_2:
; CHECK-NEXT:    l32i a2, a1, 4 # 4-byte Folded Reload
; CHECK-NEXT:    addi a8, a1, 16
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    ret

  %cond = icmp slt i32 %a, %b
  %res = zext i1 %cond to i32
  ret i32 %res
}

define i32 @f_sle(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: f_sle:
; CHECK:         addi a8, a1, -16
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    movi a8, 0
; CHECK-NEXT:    s32i a8, a1, 0 # 4-byte Folded Spill
; CHECK-NEXT:    movi a8, 1
; CHECK-NEXT:    s32i a8, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:    bge a3, a2, .LBB2_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    l32i a8, a1, 0 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB2_2:
; CHECK-NEXT:    l32i a2, a1, 4 # 4-byte Folded Reload
; CHECK-NEXT:    addi a8, a1, 16
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    ret

  %cond = icmp sle i32 %a, %b
  %res = zext i1 %cond to i32
  ret i32 %res
}

define i32 @f_sgt(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: f_sgt:
; CHECK:         addi a8, a1, -16
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    movi a8, 0
; CHECK-NEXT:    s32i a8, a1, 0 # 4-byte Folded Spill
; CHECK-NEXT:    movi a8, 1
; CHECK-NEXT:    s32i a8, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:    blt a3, a2, .LBB3_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    l32i a8, a1, 0 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB3_2:
; CHECK-NEXT:    l32i a2, a1, 4 # 4-byte Folded Reload
; CHECK-NEXT:    addi a8, a1, 16
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    ret

  %cond = icmp sgt i32 %a, %b
  %res = zext i1 %cond to i32
  ret i32 %res
}

define i32 @f_sge(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: f_sge:
; CHECK:         addi a8, a1, -16
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    movi a8, 0
; CHECK-NEXT:    s32i a8, a1, 0 # 4-byte Folded Spill
; CHECK-NEXT:    movi a8, 1
; CHECK-NEXT:    s32i a8, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:    bge a2, a3, .LBB4_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    l32i a8, a1, 0 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB4_2:
; CHECK-NEXT:    l32i a2, a1, 4 # 4-byte Folded Reload
; CHECK-NEXT:    addi a8, a1, 16
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    ret

  %cond = icmp sge i32 %a, %b
  %res = zext i1 %cond to i32
  ret i32 %res
}

define i32 @f_ne(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: f_ne:
; CHECK:         addi a8, a1, -16
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    movi a8, 0
; CHECK-NEXT:    s32i a8, a1, 0 # 4-byte Folded Spill
; CHECK-NEXT:    movi a8, 1
; CHECK-NEXT:    s32i a8, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:    bne a2, a3, .LBB5_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    l32i a8, a1, 0 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB5_2:
; CHECK-NEXT:    l32i a2, a1, 4 # 4-byte Folded Reload
; CHECK-NEXT:    addi a8, a1, 16
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    ret

  %cond = icmp ne i32 %a, %b
  %res = zext i1 %cond to i32
  ret i32 %res
}

define i32 @f_ult(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: f_ult:
; CHECK:         addi a8, a1, -16
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    movi a8, 0
; CHECK-NEXT:    s32i a8, a1, 0 # 4-byte Folded Spill
; CHECK-NEXT:    movi a8, 1
; CHECK-NEXT:    s32i a8, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:    bltu a2, a3, .LBB6_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    l32i a8, a1, 0 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB6_2:
; CHECK-NEXT:    l32i a2, a1, 4 # 4-byte Folded Reload
; CHECK-NEXT:    addi a8, a1, 16
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    ret

  %cond = icmp ult i32 %a, %b
  %res = zext i1 %cond to i32
  ret i32 %res
}

define i32 @f_ule(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: f_ule:
; CHECK:         addi a8, a1, -16
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    movi a8, 0
; CHECK-NEXT:    s32i a8, a1, 0 # 4-byte Folded Spill
; CHECK-NEXT:    movi a8, 1
; CHECK-NEXT:    s32i a8, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:    bgeu a3, a2, .LBB7_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    l32i a8, a1, 0 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB7_2:
; CHECK-NEXT:    l32i a2, a1, 4 # 4-byte Folded Reload
; CHECK-NEXT:    addi a8, a1, 16
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    ret

  %cond = icmp ule i32 %a, %b
  %res = zext i1 %cond to i32
  ret i32 %res
}

define i32 @f_ugt(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: f_ugt:
; CHECK:         addi a8, a1, -16
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    movi a8, 0
; CHECK-NEXT:    s32i a8, a1, 0 # 4-byte Folded Spill
; CHECK-NEXT:    movi a8, 1
; CHECK-NEXT:    s32i a8, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:    bltu a3, a2, .LBB8_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    l32i a8, a1, 0 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB8_2:
; CHECK-NEXT:    l32i a2, a1, 4 # 4-byte Folded Reload
; CHECK-NEXT:    addi a8, a1, 16
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    ret

  %cond = icmp ugt i32 %a, %b
  %res = zext i1 %cond to i32
  ret i32 %res
}

define i32 @f_uge(i32 %a, i32 %b) nounwind {
; CHECK-LABEL: f_uge:
; CHECK:         addi a8, a1, -16
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    movi a8, 0
; CHECK-NEXT:    s32i a8, a1, 0 # 4-byte Folded Spill
; CHECK-NEXT:    movi a8, 1
; CHECK-NEXT:    s32i a8, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:    bgeu a2, a3, .LBB9_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    l32i a8, a1, 0 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB9_2:
; CHECK-NEXT:    l32i a2, a1, 4 # 4-byte Folded Reload
; CHECK-NEXT:    addi a8, a1, 16
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    ret

  %cond = icmp uge i32 %a, %b
  %res = zext i1 %cond to i32
  ret i32 %res
}


; Tests for i64 operands

define i64 @f_eq_i64(i64 %a, i64 %b) nounwind {
; CHECK-LABEL: f_eq_i64:
; CHECK:         addi a8, a1, -16
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    # kill: def $a8 killed $a5
; CHECK-NEXT:    # kill: def $a8 killed $a4
; CHECK-NEXT:    # kill: def $a8 killed $a3
; CHECK-NEXT:    # kill: def $a8 killed $a2
; CHECK-NEXT:    xor a9, a3, a5
; CHECK-NEXT:    xor a8, a2, a4
; CHECK-NEXT:    or a8, a8, a9
; CHECK-NEXT:    movi a10, 1
; CHECK-NEXT:    movi a9, 0
; CHECK-NEXT:    s32i a9, a1, 0 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a10, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:    beq a8, a9, .LBB10_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    l32i a8, a1, 0 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB10_2:
; CHECK-NEXT:    l32i a3, a1, 0 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a2, a1, 4 # 4-byte Folded Reload
; CHECK-NEXT:    addi a8, a1, 16
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    ret

  %cond = icmp eq i64 %a, %b
  %res = zext i1 %cond to i64
  ret i64 %res
}

define i64 @f_slt_i64(i64 %a, i64 %b) nounwind {
; CHECK-LABEL: f_slt_i64:
; CHECK:         addi a8, a1, -48
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    s32i a5, a1, 12 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a4, a1, 16 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a3, a1, 20 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a2, a1, 24 # 4-byte Folded Spill
; CHECK-NEXT:    # kill: def $a8 killed $a5
; CHECK-NEXT:    # kill: def $a8 killed $a3
; CHECK-NEXT:    movi a8, 0
; CHECK-NEXT:    s32i a8, a1, 28 # 4-byte Folded Spill
; CHECK-NEXT:    movi a8, 1
; CHECK-NEXT:    s32i a8, a1, 32 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a8, a1, 36 # 4-byte Folded Spill
; CHECK-NEXT:    blt a3, a5, .LBB11_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    l32i a8, a1, 28 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 36 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB11_2:
; CHECK-NEXT:    l32i a8, a1, 24 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a9, a1, 16 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a10, a1, 32 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a11, a1, 36 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a11, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a10, a1, 8 # 4-byte Folded Spill
; CHECK-NEXT:    bltu a8, a9, .LBB11_4
; CHECK-NEXT:  # %bb.3:
; CHECK-NEXT:    l32i a8, a1, 28 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 8 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB11_4:
; CHECK-NEXT:    l32i a8, a1, 20 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a9, a1, 12 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a10, a1, 8 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a10, a1, 0 # 4-byte Folded Spill
; CHECK-NEXT:    beq a8, a9, .LBB11_6
; CHECK-NEXT:  # %bb.5:
; CHECK-NEXT:    l32i a8, a1, 4 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 0 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB11_6:
; CHECK-NEXT:    l32i a3, a1, 28 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a2, a1, 0 # 4-byte Folded Reload
; CHECK-NEXT:    addi a8, a1, 48
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    ret

  %cond = icmp slt i64 %a, %b
  %res = zext i1 %cond to i64
  ret i64 %res
}

define i64 @f_sle_i64(i64 %a, i64 %b) nounwind {
; CHECK-LABEL: f_sle_i64:
; CHECK:         addi a8, a1, -48
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    s32i a5, a1, 12 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a4, a1, 16 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a3, a1, 20 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a2, a1, 24 # 4-byte Folded Spill
; CHECK-NEXT:    # kill: def $a8 killed $a5
; CHECK-NEXT:    # kill: def $a8 killed $a3
; CHECK-NEXT:    movi a8, 0
; CHECK-NEXT:    s32i a8, a1, 28 # 4-byte Folded Spill
; CHECK-NEXT:    movi a8, 1
; CHECK-NEXT:    s32i a8, a1, 32 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a8, a1, 36 # 4-byte Folded Spill
; CHECK-NEXT:    bge a5, a3, .LBB12_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    l32i a8, a1, 28 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 36 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB12_2:
; CHECK-NEXT:    l32i a8, a1, 16 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a9, a1, 24 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a10, a1, 32 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a11, a1, 36 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a11, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a10, a1, 8 # 4-byte Folded Spill
; CHECK-NEXT:    bgeu a8, a9, .LBB12_4
; CHECK-NEXT:  # %bb.3:
; CHECK-NEXT:    l32i a8, a1, 28 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 8 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB12_4:
; CHECK-NEXT:    l32i a8, a1, 20 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a9, a1, 12 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a10, a1, 8 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a10, a1, 0 # 4-byte Folded Spill
; CHECK-NEXT:    beq a8, a9, .LBB12_6
; CHECK-NEXT:  # %bb.5:
; CHECK-NEXT:    l32i a8, a1, 4 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 0 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB12_6:
; CHECK-NEXT:    l32i a3, a1, 28 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a2, a1, 0 # 4-byte Folded Reload
; CHECK-NEXT:    addi a8, a1, 48
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    ret

  %cond = icmp sle i64 %a, %b
  %res = zext i1 %cond to i64
  ret i64 %res
}

define i64 @f_sgt_i64(i64 %a, i64 %b) nounwind {
; CHECK-LABEL: f_sgt_i64:
; CHECK:         addi a8, a1, -48
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    s32i a5, a1, 12 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a4, a1, 16 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a3, a1, 20 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a2, a1, 24 # 4-byte Folded Spill
; CHECK-NEXT:    # kill: def $a8 killed $a5
; CHECK-NEXT:    # kill: def $a8 killed $a3
; CHECK-NEXT:    movi a8, 0
; CHECK-NEXT:    s32i a8, a1, 28 # 4-byte Folded Spill
; CHECK-NEXT:    movi a8, 1
; CHECK-NEXT:    s32i a8, a1, 32 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a8, a1, 36 # 4-byte Folded Spill
; CHECK-NEXT:    blt a5, a3, .LBB13_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    l32i a8, a1, 28 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 36 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB13_2:
; CHECK-NEXT:    l32i a8, a1, 16 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a9, a1, 24 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a10, a1, 32 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a11, a1, 36 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a11, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a10, a1, 8 # 4-byte Folded Spill
; CHECK-NEXT:    bltu a8, a9, .LBB13_4
; CHECK-NEXT:  # %bb.3:
; CHECK-NEXT:    l32i a8, a1, 28 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 8 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB13_4:
; CHECK-NEXT:    l32i a8, a1, 20 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a9, a1, 12 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a10, a1, 8 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a10, a1, 0 # 4-byte Folded Spill
; CHECK-NEXT:    beq a8, a9, .LBB13_6
; CHECK-NEXT:  # %bb.5:
; CHECK-NEXT:    l32i a8, a1, 4 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 0 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB13_6:
; CHECK-NEXT:    l32i a3, a1, 28 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a2, a1, 0 # 4-byte Folded Reload
; CHECK-NEXT:    addi a8, a1, 48
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    ret

  %cond = icmp sgt i64 %a, %b
  %res = zext i1 %cond to i64
  ret i64 %res
}

define i64 @f_sge_i64(i64 %a, i64 %b) nounwind {
; CHECK-LABEL: f_sge_i64:
; CHECK:         addi a8, a1, -48
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    s32i a5, a1, 12 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a4, a1, 16 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a3, a1, 20 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a2, a1, 24 # 4-byte Folded Spill
; CHECK-NEXT:    # kill: def $a8 killed $a5
; CHECK-NEXT:    # kill: def $a8 killed $a3
; CHECK-NEXT:    movi a8, 0
; CHECK-NEXT:    s32i a8, a1, 28 # 4-byte Folded Spill
; CHECK-NEXT:    movi a8, 1
; CHECK-NEXT:    s32i a8, a1, 32 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a8, a1, 36 # 4-byte Folded Spill
; CHECK-NEXT:    bge a3, a5, .LBB14_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    l32i a8, a1, 28 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 36 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB14_2:
; CHECK-NEXT:    l32i a8, a1, 24 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a9, a1, 16 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a10, a1, 32 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a11, a1, 36 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a11, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a10, a1, 8 # 4-byte Folded Spill
; CHECK-NEXT:    bgeu a8, a9, .LBB14_4
; CHECK-NEXT:  # %bb.3:
; CHECK-NEXT:    l32i a8, a1, 28 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 8 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB14_4:
; CHECK-NEXT:    l32i a8, a1, 20 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a9, a1, 12 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a10, a1, 8 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a10, a1, 0 # 4-byte Folded Spill
; CHECK-NEXT:    beq a8, a9, .LBB14_6
; CHECK-NEXT:  # %bb.5:
; CHECK-NEXT:    l32i a8, a1, 4 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 0 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB14_6:
; CHECK-NEXT:    l32i a3, a1, 28 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a2, a1, 0 # 4-byte Folded Reload
; CHECK-NEXT:    addi a8, a1, 48
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    ret

  %cond = icmp sge i64 %a, %b
  %res = zext i1 %cond to i64
  ret i64 %res
}

define i64 @f_ne_i64(i64 %a, i64 %b) nounwind {
; CHECK-LABEL: f_ne_i64:
; CHECK:         addi a8, a1, -16
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    # kill: def $a8 killed $a5
; CHECK-NEXT:    # kill: def $a8 killed $a4
; CHECK-NEXT:    # kill: def $a8 killed $a3
; CHECK-NEXT:    # kill: def $a8 killed $a2
; CHECK-NEXT:    xor a9, a3, a5
; CHECK-NEXT:    xor a8, a2, a4
; CHECK-NEXT:    or a8, a8, a9
; CHECK-NEXT:    movi a10, 1
; CHECK-NEXT:    movi a9, 0
; CHECK-NEXT:    s32i a9, a1, 0 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a10, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:    bne a8, a9, .LBB15_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    l32i a8, a1, 0 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB15_2:
; CHECK-NEXT:    l32i a3, a1, 0 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a2, a1, 4 # 4-byte Folded Reload
; CHECK-NEXT:    addi a8, a1, 16
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    ret

  %cond = icmp ne i64 %a, %b
  %res = zext i1 %cond to i64
  ret i64 %res
}

define i64 @f_ult_i64(i64 %a, i64 %b) nounwind {
; CHECK-LABEL: f_ult_i64:
; CHECK:         addi a8, a1, -48
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    s32i a5, a1, 12 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a4, a1, 16 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a3, a1, 20 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a2, a1, 24 # 4-byte Folded Spill
; CHECK-NEXT:    # kill: def $a8 killed $a5
; CHECK-NEXT:    # kill: def $a8 killed $a3
; CHECK-NEXT:    movi a8, 0
; CHECK-NEXT:    s32i a8, a1, 28 # 4-byte Folded Spill
; CHECK-NEXT:    movi a8, 1
; CHECK-NEXT:    s32i a8, a1, 32 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a8, a1, 36 # 4-byte Folded Spill
; CHECK-NEXT:    bltu a3, a5, .LBB16_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    l32i a8, a1, 28 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 36 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB16_2:
; CHECK-NEXT:    l32i a8, a1, 24 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a9, a1, 16 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a10, a1, 32 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a11, a1, 36 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a11, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a10, a1, 8 # 4-byte Folded Spill
; CHECK-NEXT:    bltu a8, a9, .LBB16_4
; CHECK-NEXT:  # %bb.3:
; CHECK-NEXT:    l32i a8, a1, 28 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 8 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB16_4:
; CHECK-NEXT:    l32i a8, a1, 20 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a9, a1, 12 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a10, a1, 8 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a10, a1, 0 # 4-byte Folded Spill
; CHECK-NEXT:    beq a8, a9, .LBB16_6
; CHECK-NEXT:  # %bb.5:
; CHECK-NEXT:    l32i a8, a1, 4 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 0 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB16_6:
; CHECK-NEXT:    l32i a3, a1, 28 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a2, a1, 0 # 4-byte Folded Reload
; CHECK-NEXT:    addi a8, a1, 48
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    ret

  %cond = icmp ult i64 %a, %b
  %res = zext i1 %cond to i64
  ret i64 %res
}

define i64 @f_ule_i64(i64 %a, i64 %b) nounwind {
; CHECK-LABEL: f_ule_i64:
; CHECK:         addi a8, a1, -48
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    s32i a5, a1, 12 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a4, a1, 16 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a3, a1, 20 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a2, a1, 24 # 4-byte Folded Spill
; CHECK-NEXT:    # kill: def $a8 killed $a5
; CHECK-NEXT:    # kill: def $a8 killed $a3
; CHECK-NEXT:    movi a8, 0
; CHECK-NEXT:    s32i a8, a1, 28 # 4-byte Folded Spill
; CHECK-NEXT:    movi a8, 1
; CHECK-NEXT:    s32i a8, a1, 32 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a8, a1, 36 # 4-byte Folded Spill
; CHECK-NEXT:    bgeu a5, a3, .LBB17_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    l32i a8, a1, 28 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 36 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB17_2:
; CHECK-NEXT:    l32i a8, a1, 16 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a9, a1, 24 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a10, a1, 32 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a11, a1, 36 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a11, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a10, a1, 8 # 4-byte Folded Spill
; CHECK-NEXT:    bgeu a8, a9, .LBB17_4
; CHECK-NEXT:  # %bb.3:
; CHECK-NEXT:    l32i a8, a1, 28 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 8 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB17_4:
; CHECK-NEXT:    l32i a8, a1, 20 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a9, a1, 12 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a10, a1, 8 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a10, a1, 0 # 4-byte Folded Spill
; CHECK-NEXT:    beq a8, a9, .LBB17_6
; CHECK-NEXT:  # %bb.5:
; CHECK-NEXT:    l32i a8, a1, 4 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 0 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB17_6:
; CHECK-NEXT:    l32i a3, a1, 28 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a2, a1, 0 # 4-byte Folded Reload
; CHECK-NEXT:    addi a8, a1, 48
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    ret

  %cond = icmp ule i64 %a, %b
  %res = zext i1 %cond to i64
  ret i64 %res
}

define i64 @f_ugt_i64(i64 %a, i64 %b) nounwind {
; CHECK-LABEL: f_ugt_i64:
; CHECK:         addi a8, a1, -48
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    s32i a5, a1, 12 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a4, a1, 16 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a3, a1, 20 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a2, a1, 24 # 4-byte Folded Spill
; CHECK-NEXT:    # kill: def $a8 killed $a5
; CHECK-NEXT:    # kill: def $a8 killed $a3
; CHECK-NEXT:    movi a8, 0
; CHECK-NEXT:    s32i a8, a1, 28 # 4-byte Folded Spill
; CHECK-NEXT:    movi a8, 1
; CHECK-NEXT:    s32i a8, a1, 32 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a8, a1, 36 # 4-byte Folded Spill
; CHECK-NEXT:    bltu a5, a3, .LBB18_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    l32i a8, a1, 28 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 36 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB18_2:
; CHECK-NEXT:    l32i a8, a1, 16 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a9, a1, 24 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a10, a1, 32 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a11, a1, 36 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a11, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a10, a1, 8 # 4-byte Folded Spill
; CHECK-NEXT:    bltu a8, a9, .LBB18_4
; CHECK-NEXT:  # %bb.3:
; CHECK-NEXT:    l32i a8, a1, 28 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 8 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB18_4:
; CHECK-NEXT:    l32i a8, a1, 20 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a9, a1, 12 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a10, a1, 8 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a10, a1, 0 # 4-byte Folded Spill
; CHECK-NEXT:    beq a8, a9, .LBB18_6
; CHECK-NEXT:  # %bb.5:
; CHECK-NEXT:    l32i a8, a1, 4 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 0 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB18_6:
; CHECK-NEXT:    l32i a3, a1, 28 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a2, a1, 0 # 4-byte Folded Reload
; CHECK-NEXT:    addi a8, a1, 48
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    ret

  %cond = icmp ugt i64 %a, %b
  %res = zext i1 %cond to i64
  ret i64 %res
}

define i64 @f_uge_i64(i64 %a, i64 %b) nounwind {
; CHECK-LABEL: f_uge_i64:
; CHECK:         addi a8, a1, -48
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    s32i a5, a1, 12 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a4, a1, 16 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a3, a1, 20 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a2, a1, 24 # 4-byte Folded Spill
; CHECK-NEXT:    # kill: def $a8 killed $a5
; CHECK-NEXT:    # kill: def $a8 killed $a3
; CHECK-NEXT:    movi a8, 0
; CHECK-NEXT:    s32i a8, a1, 28 # 4-byte Folded Spill
; CHECK-NEXT:    movi a8, 1
; CHECK-NEXT:    s32i a8, a1, 32 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a8, a1, 36 # 4-byte Folded Spill
; CHECK-NEXT:    bgeu a3, a5, .LBB19_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    l32i a8, a1, 28 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 36 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB19_2:
; CHECK-NEXT:    l32i a8, a1, 24 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a9, a1, 16 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a10, a1, 32 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a11, a1, 36 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a11, a1, 4 # 4-byte Folded Spill
; CHECK-NEXT:    s32i a10, a1, 8 # 4-byte Folded Spill
; CHECK-NEXT:    bgeu a8, a9, .LBB19_4
; CHECK-NEXT:  # %bb.3:
; CHECK-NEXT:    l32i a8, a1, 28 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 8 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB19_4:
; CHECK-NEXT:    l32i a8, a1, 20 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a9, a1, 12 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a10, a1, 8 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a10, a1, 0 # 4-byte Folded Spill
; CHECK-NEXT:    beq a8, a9, .LBB19_6
; CHECK-NEXT:  # %bb.5:
; CHECK-NEXT:    l32i a8, a1, 4 # 4-byte Folded Reload
; CHECK-NEXT:    s32i a8, a1, 0 # 4-byte Folded Spill
; CHECK-NEXT:  .LBB19_6:
; CHECK-NEXT:    l32i a3, a1, 28 # 4-byte Folded Reload
; CHECK-NEXT:    l32i a2, a1, 0 # 4-byte Folded Reload
; CHECK-NEXT:    addi a8, a1, 48
; CHECK-NEXT:    or a1, a8, a8
; CHECK-NEXT:    ret

  %cond = icmp uge i64 %a, %b
  %res = zext i1 %cond to i64
  ret i64 %res
}
