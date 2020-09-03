;; Test that (mul (add x, c1), c2) can be transformed to
;; (add (mul x, c2), c1*c2) if profitable.

; RUN: llc -mtriple=riscv32 -mattr=+m -verify-machineinstrs < %s \
; RUN:   | FileCheck -check-prefix=RV32IM %s
; RUN: llc -mtriple=riscv64 -mattr=+m -verify-machineinstrs < %s \
; RUN:   | FileCheck -check-prefix=RV64IM %s

define signext i32 @add_mul_trans_accept_1(i32 %x) {
; RV32IM-LABEL: add_mul_trans_accept_1
; RV32IM:       # %bb.0:
; RV32IM-NEXT:    addi a1, zero, 11
; RV32IM-NEXT:    mul a0, a0, a1
; RV32IM-NEXT:    addi a0, a0, 407
; RV32IM-NEXT:    ret
;
; RV64IM-LABEL: add_mul_trans_accept_1
; RV64IM:       # %bb.0:
; RV64IM-NEXT:    addi a1, zero, 11
; RV64IM-NEXT:    mul a0, a0, a1
; RV64IM-NEXT:    addiw a0, a0, 407
; RV64IM-NEXT:    ret
  %tmp0 = add i32 %x, 37
  %tmp1 = mul i32 %tmp0, 11
  ret i32 %tmp1
}

define signext i32 @add_mul_trans_accept_2(i32 %x) {
; RV32IM-LABEL: add_mul_trans_accept_2
; RV32IM:       # %bb.0:
; RV32IM-NEXT:    addi a1, zero, 13
; RV32IM-NEXT:    mul a0, a0, a1
; RV32IM-NEXT:    lui a1, 28
; RV32IM-NEXT:    addi a1, a1, 1701
; RV32IM-NEXT:    add a0, a0, a1
; RV32IM-NEXT:    ret
;
; RV64IM-LABEL: add_mul_trans_accept_2
; RV64IM:       # %bb.0:
; RV64IM-NEXT:    addi a1, zero, 13
; RV64IM-NEXT:    mul a0, a0, a1
; RV64IM-NEXT:    lui a1, 28
; RV64IM-NEXT:    addiw a1, a1, 1701
; RV64IM-NEXT:    addw a0, a0, a1
; RV64IM-NEXT:    ret
  %tmp0 = add i32 %x, 8953
  %tmp1 = mul i32 %tmp0, 13
  ret i32 %tmp1
}

define signext i32 @add_mul_trans_reject_1(i32 %x) {
; RV32IM-LABEL: add_mul_trans_reject_1
; RV32IM:       # %bb.0:
; RV32IM-NEXT:    addi a1, zero, 19
; RV32IM-NEXT:    mul a0, a0, a1
; RV32IM-NEXT:    lui a1, 9
; RV32IM-NEXT:    addi a1, a1, 585
; RV32IM-NEXT:    add a0, a0, a1
; RV32IM-NEXT:    ret
;
; RV64IM-LABEL: add_mul_trans_reject_1
; RV64IM:       # %bb.0:
; RV64IM-NEXT:    addi a1, zero, 19
; RV64IM-NEXT:    mul a0, a0, a1
; RV64IM-NEXT:    lui a1, 9
; RV64IM-NEXT:    addiw a1, a1, 585
; RV64IM-NEXT:    addw a0, a0, a1
; RV64IM-NEXT:    ret
  %tmp0 = add i32 %x, 1971
  %tmp1 = mul i32 %tmp0, 19
  ret i32 %tmp1
}

define signext i32 @add_mul_trans_reject_2(i32 %x) {
; RV32IM:       # %bb.0:
; RV32IM-NEXT:    lui a1, 792
; RV32IM-NEXT:    addi a1, a1, -1709
; RV32IM-NEXT:    mul a0, a0, a1
; RV32IM-NEXT:    lui a1, 1014660
; RV32IM-NEXT:    addi a1, a1, -1891
; RV32IM-NEXT:    add a0, a0, a1
; RV32IM-NEXT:    ret
;
; RV64IM:       # %bb.0:
; RV64IM-NEXT:    lui a1, 792
; RV64IM-NEXT:    addiw a1, a1, -1709
; RV64IM-NEXT:    mul a0, a0, a1
; RV64IM-NEXT:    lui a1, 1014660
; RV64IM-NEXT:    addiw a1, a1, -1891
; RV64IM-NEXT:    addw a0, a0, a1
; RV64IM-NEXT:    ret
  %tmp0 = add i32 %x, 1841231
  %tmp1 = mul i32 %tmp0, 3242323
  ret i32 %tmp1
}
