; RUN: llc -mtriple=riscv32 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=RV32I

; Register-immediate instructions

define i32 @addi(i32 %a) nounwind {
; RV32I-LABEL: addi:
; RV32I: addi a0, a0, 1
; RV32I: jalr zero, ra, 0
; TODO: check support for materialising larger constants
  %1 = add i32 %a, 1
  ret i32 %1
}

define i32 @slti(i32 %a) nounwind {
; RV32I-LABEL: slti:
; RV32I: slti a0, a0, 2
; RV32I: jalr zero, ra, 0
  %1 = icmp slt i32 %a, 2
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @sltiu(i32 %a) nounwind {
; RV32I-LABEL: sltiu:
; RV32I: sltiu a0, a0, 3
; RV32I: jalr zero, ra, 0
  %1 = icmp ult i32 %a, 3
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @xori(i32 %a) nounwind {
; RV32I-LABEL: xori:
; RV32I: xori a0, a0, 4
; RV32I: jalr zero, ra, 0
  %1 = xor i32 %a, 4
  ret i32 %1
}

define i32 @ori(i32 %a) nounwind {
; RV32I-LABEL: ori:
; RV32I: ori a0, a0, 5
; RV32I: jalr zero, ra, 0
  %1 = or i32 %a, 5
  ret i32 %1
}

define i32 @andi(i32 %a) nounwind {
; RV32I-LABEL: andi:
; RV32I: andi a0, a0, 6
; RV32I: jalr zero, ra, 0
  %1 = and i32 %a, 6
  ret i32 %1
}

define i32 @slli(i32 %a) nounwind {
; RV32I-LABEL: slli:
; RV32I: slli a0, a0, 7
; RV32I: jalr zero, ra, 0
  %1 = shl i32 %a, 7
  ret i32 %1
}

define i32 @srli(i32 %a) nounwind {
; RV32I-LABEL: srli:
; RV32I: srli a0, a0, 8
; RV32I: jalr zero, ra, 0
  %1 = lshr i32 %a, 8
  ret i32 %1
}

define i32 @srai(i32 %a) nounwind {
; RV32I-LABEL: srai:
; RV32I: srai a0, a0, 9
; RV32I: jalr zero, ra, 0
  %1 = ashr i32 %a, 9
  ret i32 %1
}

; Register-register instructions

define i32 @add(i32 %a, i32 %b) nounwind {
; RV32I-LABEL: add:
; RV32I: add a0, a0, a1
; RV32I: jalr zero, ra, 0
  %1 = add i32 %a, %b
  ret i32 %1
}

define i32 @sub(i32 %a, i32 %b) nounwind {
; RV32I-LABEL: sub:
; RV32I: sub a0, a0, a1
; RV32I: jalr zero, ra, 0
  %1 = sub i32 %a, %b
  ret i32 %1
}

define i32 @sll(i32 %a, i32 %b) nounwind {
; RV32I-LABEL: sll:
; RV32I: sll a0, a0, a1
; RV32I: jalr zero, ra, 0
  %1 = shl i32 %a, %b
  ret i32 %1
}

define i32 @slt(i32 %a, i32 %b) nounwind {
; RV32I-LABEL: slt:
; RV32I: slt a0, a0, a1
; RV32I: jalr zero, ra, 0
  %1 = icmp slt i32 %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @sltu(i32 %a, i32 %b) nounwind {
; RV32I-LABEL: sltu:
; RV32I: sltu a0, a0, a1
; RV32I: jalr zero, ra, 0
  %1 = icmp ult i32 %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @xor(i32 %a, i32 %b) nounwind {
; RV32I-LABEL: xor:
; RV32I: xor a0, a0, a1
; RV32I: jalr zero, ra, 0
  %1 = xor i32 %a, %b
  ret i32 %1
}

define i32 @srl(i32 %a, i32 %b) nounwind {
; RV32I-LABEL: srl:
; RV32I: srl a0, a0, a1
; RV32I: jalr zero, ra, 0
  %1 = lshr i32 %a, %b
  ret i32 %1
}

define i32 @sra(i32 %a, i32 %b) nounwind {
; RV32I-LABEL: sra:
; RV32I: sra a0, a0, a1
; RV32I: jalr zero, ra, 0
  %1 = ashr i32 %a, %b
  ret i32 %1
}

define i32 @or(i32 %a, i32 %b) nounwind {
; RV32I-LABEL: or:
; RV32I: or a0, a0, a1
; RV32I: jalr zero, ra, 0
  %1 = or i32 %a, %b
  ret i32 %1
}

define i32 @and(i32 %a, i32 %b) nounwind {
; RV32I-LABEL: and:
; RV32I: and a0, a0, a1
; RV32I: jalr zero, ra, 0
  %1 = and i32 %a, %b
  ret i32 %1
}
