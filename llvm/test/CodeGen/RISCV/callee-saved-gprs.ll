; RUN: llc -mtriple=riscv32 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=RV32I
; RUN: llc -mtriple=riscv32 -verify-machineinstrs -frame-pointer=all < %s \
; RUN:   | FileCheck %s -check-prefix=RV32I-WITH-FP
; RUN: llc -mtriple=riscv64 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=RV64I
; RUN: llc -mtriple=riscv64 -verify-machineinstrs -frame-pointer=all < %s \
; RUN:   | FileCheck %s -check-prefix=RV64I-WITH-FP

@var = global [32 x i32] zeroinitializer

define void @foo() {
; RV32I-LABEL: foo:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi sp, sp, -80
; RV32I-NEXT:    sw s0, 76(sp)
; RV32I-NEXT:    sw s1, 72(sp)
; RV32I-NEXT:    sw s2, 68(sp)
; RV32I-NEXT:    sw s3, 64(sp)
; RV32I-NEXT:    sw s4, 60(sp)
; RV32I-NEXT:    sw s5, 56(sp)
; RV32I-NEXT:    sw s6, 52(sp)
; RV32I-NEXT:    sw s7, 48(sp)
; RV32I-NEXT:    sw s8, 44(sp)
; RV32I-NEXT:    sw s9, 40(sp)
; RV32I-NEXT:    sw s10, 36(sp)
; RV32I-NEXT:    sw s11, 32(sp)
; RV32I-NEXT:    lui a0, %hi(var)
; RV32I-NEXT:    addi a1, a0, %lo(var)
;
; RV32I-WITH-FP-LABEL: foo:
; RV32I-WITH-FP:       # %bb.0:
; RV32I-WITH-FP-NEXT:    addi sp, sp, -80
; RV32I-WITH-FP-NEXT:    sw ra, 76(sp)
; RV32I-WITH-FP-NEXT:    sw s0, 72(sp)
; RV32I-WITH-FP-NEXT:    sw s1, 68(sp)
; RV32I-WITH-FP-NEXT:    sw s2, 64(sp)
; RV32I-WITH-FP-NEXT:    sw s3, 60(sp)
; RV32I-WITH-FP-NEXT:    sw s4, 56(sp)
; RV32I-WITH-FP-NEXT:    sw s5, 52(sp)
; RV32I-WITH-FP-NEXT:    sw s6, 48(sp)
; RV32I-WITH-FP-NEXT:    sw s7, 44(sp)
; RV32I-WITH-FP-NEXT:    sw s8, 40(sp)
; RV32I-WITH-FP-NEXT:    sw s9, 36(sp)
; RV32I-WITH-FP-NEXT:    sw s10, 32(sp)
; RV32I-WITH-FP-NEXT:    sw s11, 28(sp)
; RV32I-WITH-FP-NEXT:    addi s0, sp, 80
; RV32I-WITH-FP-NEXT:    lui a0, %hi(var)
; RV32I-WITH-FP-NEXT:    addi a1, a0, %lo(var)
;
; RV64I-LABEL: foo:
; RV64I:       # %bb.0:
; RV64I-NEXT:    addi sp, sp, -144
; RV64I-NEXT:    sd s0, 136(sp)
; RV64I-NEXT:    sd s1, 128(sp)
; RV64I-NEXT:    sd s2, 120(sp)
; RV64I-NEXT:    sd s3, 112(sp)
; RV64I-NEXT:    sd s4, 104(sp)
; RV64I-NEXT:    sd s5, 96(sp)
; RV64I-NEXT:    sd s6, 88(sp)
; RV64I-NEXT:    sd s7, 80(sp)
; RV64I-NEXT:    sd s8, 72(sp)
; RV64I-NEXT:    sd s9, 64(sp)
; RV64I-NEXT:    sd s10, 56(sp)
; RV64I-NEXT:    sd s11, 48(sp)
; RV64I-NEXT:    lui a0, %hi(var)
; RV64I-NEXT:    addi a1, a0, %lo(var)
;
; RV64I-WITH-FP-LABEL: foo:
; RV64I-WITH-FP:       # %bb.0:
; RV64I-WITH-FP-NEXT:    addi sp, sp, -160
; RV64I-WITH-FP-NEXT:    sd ra, 152(sp)
; RV64I-WITH-FP-NEXT:    sd s0, 144(sp)
; RV64I-WITH-FP-NEXT:    sd s1, 136(sp)
; RV64I-WITH-FP-NEXT:    sd s2, 128(sp)
; RV64I-WITH-FP-NEXT:    sd s3, 120(sp)
; RV64I-WITH-FP-NEXT:    sd s4, 112(sp)
; RV64I-WITH-FP-NEXT:    sd s5, 104(sp)
; RV64I-WITH-FP-NEXT:    sd s6, 96(sp)
; RV64I-WITH-FP-NEXT:    sd s7, 88(sp)
; RV64I-WITH-FP-NEXT:    sd s8, 80(sp)
; RV64I-WITH-FP-NEXT:    sd s9, 72(sp)
; RV64I-WITH-FP-NEXT:    sd s10, 64(sp)
; RV64I-WITH-FP-NEXT:    sd s11, 56(sp)
; RV64I-WITH-FP-NEXT:    addi s0, sp, 160
; RV64I-WITH-FP-NEXT:    lui a0, %hi(var)
; RV64I-WITH-FP-NEXT:    addi a1, a0, %lo(var)
  %val = load [32 x i32], [32 x i32]* @var
  store volatile [32 x i32] %val, [32 x i32]* @var
  ret void
}
