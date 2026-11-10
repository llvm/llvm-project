; RUN: llc -mtriple=bpfel -mcpu=v5 -verify-machineinstrs -show-mc-encoding < %s | FileCheck %s --check-prefixes=V5,I32
; RUN: llc -mtriple=bpfel -mcpu=v3 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefixes=NO-HMUL,I32 --implicit-check-not="uh*=" --implicit-check-not="sh*="
; RUN: llc -mtriple=bpfel -mcpu=v4 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefixes=NO-HMUL,I32 --implicit-check-not="uh*=" --implicit-check-not="sh*="
; RUN: llc -mtriple=bpfel -mcpu=v5 -disable-hmul -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefixes=NO-HMUL,I32 --implicit-check-not="uh*=" --implicit-check-not="sh*="

declare i64 @llvm.umulh.i64(i64, i64)
declare i64 @llvm.smulh.i64(i64, i64)
declare i32 @llvm.umulh.i32(i32, i32)
declare i32 @llvm.smulh.i32(i32, i32)

; --- 64-bit high-half multiply instruction selection ---
; Select UHMUL/SHMUL on v5, covering register operands and immediate boundaries.

define i64 @uhmul_reg(i64 %a, i64 %b) {
; V5-LABEL: uhmul_reg:
; V5:       r0 = r1
; V5-NEXT:  r0 uh*= r2                              # encoding: [0x2f,0x20,0x01,0x00,0x00,0x00,0x00,0x00]
; V5-NEXT:  exit
; NO-HMUL-LABEL: uhmul_reg:
; NO-HMUL:       exit
  %result = call i64 @llvm.umulh.i64(i64 %a, i64 %b)
  ret i64 %result
}

define i64 @uhmul_imm_neg2(i64 %a) {
; V5-LABEL: uhmul_imm_neg2:
; V5:       r0 = r1
; V5-NEXT:  r0 uh*= -2                              # encoding: [0x27,0x00,0x01,0x00,0xfe,0xff,0xff,0xff]
; V5-NEXT:  exit
; NO-HMUL-LABEL: uhmul_imm_neg2:
; NO-HMUL:       exit
  %result = call i64 @llvm.umulh.i64(i64 %a, i64 -2)
  ret i64 %result
}

; The immediate operand is sign-extended from 32 to 64 bits.
define i64 @uhmul_imm_s32_max(i64 %a) {
; V5-LABEL: uhmul_imm_s32_max:
; V5:       r0 uh*= 2147483647
; V5-NEXT:  exit
; NO-HMUL-LABEL: uhmul_imm_s32_max:
; NO-HMUL:       exit
  %result = call i64 @llvm.umulh.i64(i64 %a, i64 2147483647)
  ret i64 %result
}

define i64 @uhmul_imm_s32_min(i64 %a) {
; V5-LABEL: uhmul_imm_s32_min:
; V5:       r0 uh*= -2147483648
; V5-NEXT:  exit
; NO-HMUL-LABEL: uhmul_imm_s32_min:
; NO-HMUL:       exit
  %result = call i64 @llvm.umulh.i64(i64 %a, i64 -2147483648)
  ret i64 %result
}

; 0x00000000ffffffff cannot be represented by a sign-extended imm32.
; Materialize the full value and use the register form instead of imm32 -1.
define i64 @uhmul_imm_u32_max(i64 %a) {
; V5-LABEL: uhmul_imm_u32_max:
; V5:       r{{[0-9]+}} = 4294967295 ll
; V5:       r0 uh*= r{{[0-9]+}}
; V5-NEXT:  exit
; NO-HMUL-LABEL: uhmul_imm_u32_max:
; NO-HMUL:       exit
  %result = call i64 @llvm.umulh.i64(i64 %a, i64 4294967295)
  ret i64 %result
}

define i64 @shmul_reg(i64 %a, i64 %b) {
; V5-LABEL: shmul_reg:
; V5:       r0 = r1
; V5-NEXT:  r0 sh*= r2                              # encoding: [0x2f,0x20,0x02,0x00,0x00,0x00,0x00,0x00]
; V5-NEXT:  exit
; NO-HMUL-LABEL: shmul_reg:
; NO-HMUL:       exit
  %result = call i64 @llvm.smulh.i64(i64 %a, i64 %b)
  ret i64 %result
}

define i64 @shmul_imm_neg2(i64 %a) {
; V5-LABEL: shmul_imm_neg2:
; V5:       r0 = r1
; V5-NEXT:  r0 sh*= -2                              # encoding: [0x27,0x00,0x02,0x00,0xfe,0xff,0xff,0xff]
; V5-NEXT:  exit
; NO-HMUL-LABEL: shmul_imm_neg2:
; NO-HMUL:       exit
  %result = call i64 @llvm.smulh.i64(i64 %a, i64 -2)
  ret i64 %result
}

; The immediate operand is sign-extended from 32 to 64 bits.
define i64 @shmul_imm_s32_max(i64 %a) {
; V5-LABEL: shmul_imm_s32_max:
; V5:       r0 sh*= 2147483647
; V5-NEXT:  exit
; NO-HMUL-LABEL: shmul_imm_s32_max:
; NO-HMUL:       exit
  %result = call i64 @llvm.smulh.i64(i64 %a, i64 2147483647)
  ret i64 %result
}

define i64 @shmul_imm_s32_min(i64 %a) {
; V5-LABEL: shmul_imm_s32_min:
; V5:       r0 sh*= -2147483648
; V5-NEXT:  exit
; NO-HMUL-LABEL: shmul_imm_s32_min:
; NO-HMUL:       exit
  %result = call i64 @llvm.smulh.i64(i64 %a, i64 -2147483648)
  ret i64 %result
}

; 0x00000000ffffffff cannot be represented by a sign-extended imm32.
; Materialize the full value and use the register form instead of imm32 -1.
define i64 @shmul_imm_u32_max(i64 %a) {
; V5-LABEL: shmul_imm_u32_max:
; V5:       r{{[0-9]+}} = 4294967295 ll
; V5:       r0 sh*= r{{[0-9]+}}
; V5-NEXT:  exit
; NO-HMUL-LABEL: shmul_imm_u32_max:
; NO-HMUL:       exit
  %result = call i64 @llvm.smulh.i64(i64 %a, i64 4294967295)
  ret i64 %result
}

; --- 32-bit high-half multiply expansion ---
; There are no ALU32 high-half multiply instructions, even on v5.
; Expand i32 MULHU/MULHS to a full i64 product and extract bits 32..63.

define i32 @uhmul_i32(i32 %a, i32 %b) {
; I32-LABEL: uhmul_i32:
; I32:       r0 *= r2
; I32-NEXT:  r0 >>= 32
; I32:       exit
  %result = call i32 @llvm.umulh.i32(i32 %a, i32 %b)
  ret i32 %result
}

define i32 @shmul_i32(i32 %a, i32 %b) {
; I32-LABEL: shmul_i32:
; I32:       r0 *= r2
; I32-NEXT:  r0 >>= 32
; I32:       exit
  %result = call i32 @llvm.smulh.i32(i32 %a, i32 %b)
  ret i32 %result
}

; --- Low-half multiply regression ---
; Ordinary multiplication must still select MUL with offset 0, including on v5.

define i64 @mul_low(i64 %a, i64 %b) {
; V5-LABEL: mul_low:
; V5:       r0 = r1
; V5-NEXT:  r0 *= r2                                # encoding: [0x2f,0x20,0x00,0x00,0x00,0x00,0x00,0x00]
; V5-NEXT:  exit
; NO-HMUL-LABEL: mul_low:
; NO-HMUL:       r0 *= r2
  %product = mul i64 %a, %b
  ret i64 %product
}
