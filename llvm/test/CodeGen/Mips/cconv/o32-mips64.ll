; RUN: llc -mtriple=mips64 -target-abi=o32 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=BE,HARD
; RUN: llc -mtriple=mips64el -target-abi=o32 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=LE,HARD
; RUN: llc -mtriple=mips64 -target-abi=o32 -mattr=+soft-float -verify-machineinstrs < %s | FileCheck %s --check-prefixes=BE,SOFT-BE
; RUN: llc -mtriple=mips64el -target-abi=o32 -mattr=+soft-float -verify-machineinstrs < %s | FileCheck %s --check-prefixes=LE,SOFT-LE
; RUN: llc -mtriple=mips64 -mcpu=mips64r2 -target-abi=o32 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=ALL,O32-BE
; RUN: llc -mtriple=mips64el -mcpu=mips64r2 -target-abi=o32 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=ALL,O32-LE
; RUN: llc -mtriple=mips64 -mcpu=mips64r2 -target-abi=n32 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=ALL,NABI
; RUN: llc -mtriple=mips64el -mcpu=mips64r2 -target-abi=n32 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=ALL,NABI
; RUN: llc -mtriple=mips64 -mcpu=mips64r2 -target-abi=n64 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=ALL,NABI
; RUN: llc -mtriple=mips64el -mcpu=mips64r2 -target-abi=n64 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=ALL,NABI

; Selecting O32's GPR mode must preserve the CPU's revision 2 instructions
; and 64-bit FPR mode. N32/N64 must continue to use 64-bit GPR operations.
; O32-BE: .module fp=64
; O32-LE: .module fp=64

declare i32 @llvm.fshr.i32(i32, i32, i32)

define i32 @rotate(i32 %value) {
; ALL-LABEL: rotate:
; O32-BE: rotr $2, $4, 7
; O32-LE: rotr $2, $4, 7
; NABI: sll [[WORD:\$[0-9]+]], $4, 0
; NABI: rotr $2, [[WORD]], 7
  %result = call i32 @llvm.fshr.i32(i32 %value, i32 %value, i32 7)
  ret i32 %result
}

define double @add_double(double %x, double %y) {
; ALL-LABEL: add_double:
; O32-BE: add.d $f0, $f12, $f14
; O32-LE: add.d $f0, $f12, $f14
; NABI: add.d $f0, $f12, $f13
  %result = fadd double %x, %y
  ret double %result
}

define i64 @increment(i64 %value) {
; ALL-LABEL: increment:
; O32-BE: addiu $3, $5, 1
; O32-BE: sltiu $1, $3, 1
; O32-BE: addu $2, $4, $1
; O32-LE: addiu $2, $4, 1
; O32-LE: sltiu $1, $2, 1
; O32-LE: addu $3, $5, $1
; NABI: daddiu $2, $4, 1
  %result = add i64 %value, 1
  ret i64 %result
}

; O32 passes and returns i64 in pairs of 32-bit registers even on a 64-bit
; CPU. The caller and callee must agree on the pieces.
define i64 @identity_i64(i64 %value) {
; BE-LABEL: identity_i64:
; BE:       # %bb.0:
; BE-NEXT:    move $2, $4
; BE-NEXT:    jr $ra
; BE-NEXT:    move $3, $5
;
; LE-LABEL: identity_i64:
; LE:       # %bb.0:
; LE-NEXT:    move $2, $4
; LE-NEXT:    jr $ra
; LE-NEXT:    move $3, $5
  ret i64 %value
}

declare i64 @get_i64()

define i64 @call_i64() {
; BE-LABEL: call_i64:
; BE:       # %bb.0:
; BE-NEXT:    addiu $sp, $sp, -24
; BE-NEXT:    .cfi_def_cfa_offset 24
; BE-NEXT:    sw $ra, 20($sp) # 4-byte Folded Spill
; BE-NEXT:    .cfi_offset 31, -4
; BE-NEXT:    jal get_i64
; BE-NEXT:    nop
; BE-NEXT:    addiu $3, $3, 1
; BE-NEXT:    sltiu $1, $3, 1
; BE-NEXT:    addu $2, $2, $1
; BE-NEXT:    lw $ra, 20($sp) # 4-byte Folded Reload
; BE-NEXT:    jr $ra
; BE-NEXT:    addiu $sp, $sp, 24
;
; LE-LABEL: call_i64:
; LE:       # %bb.0:
; LE-NEXT:    addiu $sp, $sp, -24
; LE-NEXT:    .cfi_def_cfa_offset 24
; LE-NEXT:    sw $ra, 20($sp) # 4-byte Folded Spill
; LE-NEXT:    .cfi_offset 31, -4
; LE-NEXT:    jal get_i64
; LE-NEXT:    nop
; LE-NEXT:    addiu $2, $2, 1
; LE-NEXT:    sltiu $1, $2, 1
; LE-NEXT:    addu $3, $3, $1
; LE-NEXT:    lw $ra, 20($sp) # 4-byte Folded Reload
; LE-NEXT:    jr $ra
; LE-NEXT:    addiu $sp, $sp, 24
  %value = call i64 @get_i64()
  %result = add i64 %value, 1
  ret i64 %result
}

; Soft-float doubles use the same integer register pairs.
define double @identity_double(double %value) {
; HARD-LABEL: identity_double:
; HARD:       # %bb.0:
; HARD-NEXT:    jr $ra
; HARD-NEXT:    mov.d $f0, $f12
;
; SOFT-BE-LABEL: identity_double:
; SOFT-BE:       # %bb.0:
; SOFT-BE-NEXT:    move $2, $4
; SOFT-BE-NEXT:    jr $ra
; SOFT-BE-NEXT:    move $3, $5
;
; SOFT-LE-LABEL: identity_double:
; SOFT-LE:       # %bb.0:
; SOFT-LE-NEXT:    move $2, $4
; SOFT-LE-NEXT:    jr $ra
; SOFT-LE-NEXT:    move $3, $5
  ret double %value
}
