; RUN: llc -mtriple=mipsel-linux-gnu -mcpu=mips1 -relocation-model=static \
; RUN:   -verify-machineinstrs < %s | FileCheck %s -check-prefixes=ALL,LE \
; RUN:   --implicit-check-not=sdc1 --implicit-check-not=ldc1
; RUN: llc -mtriple=mips-linux-gnu -mcpu=mips1 -relocation-model=static \
; RUN:   -verify-machineinstrs < %s | FileCheck %s -check-prefixes=ALL,BE \
; RUN:   --implicit-check-not=sdc1 --implicit-check-not=ldc1

; MIPS I has no sdc1/ldc1. Double precision registers are saved, spilled and
; reloaded with two swc1/lwc1, using the same memory layout as sdc1/ldc1.

declare void @g()
declare void @h(ptr)

; Callee-saved $d10 ($f20/$f21). The .cfi_offset directives come after all
; of the saves.
define double @csr(double %a, double %b) {
; ALL-LABEL: csr:
; ALL:         addiu $sp, $sp, -32
; LE-NEXT:     .cfi_def_cfa_offset 32
; LE-NEXT:     swc1 $f20, 24($sp)
; LE-NEXT:     swc1 $f21, 28($sp)
; LE-NEXT:     sw $ra, 20($sp)
; LE-NEXT:     .cfi_offset 52, -8
; LE-NEXT:     .cfi_offset 53, -4
; LE-NEXT:     .cfi_offset 31, -12
; BE-NEXT:     .cfi_def_cfa_offset 32
; BE-NEXT:     swc1 $f21, 24($sp)
; BE-NEXT:     swc1 $f20, 28($sp)
; BE-NEXT:     sw $ra, 20($sp)
; BE-NEXT:     .cfi_offset 53, -8
; BE-NEXT:     .cfi_offset 52, -4
; BE-NEXT:     .cfi_offset 31, -12
; ALL:         jal g
; ALL:         lw $ra, 20($sp)
; LE-NEXT:     lwc1 $f20, 24($sp)
; LE-NEXT:     lwc1 $f21, 28($sp)
; BE-NEXT:     lwc1 $f21, 24($sp)
; BE-NEXT:     lwc1 $f20, 28($sp)
; ALL-NEXT:    jr $ra
  %x = fadd double %a, %b
  call void @g()
  %y = fmul double %x, %a
  ret double %y
}

; With a frame pointer, $sp is restored before the first restore.
define double @csr_fp(double %a, double %b, i32 %n) {
; ALL-LABEL: csr_fp:
; ALL:         move $sp, $fp
; ALL-NEXT:    lw $fp, 8($sp)
; ALL-NEXT:    lw $ra, 12($sp)
; LE-NEXT:     lwc1 $f20, 16($sp)
; LE-NEXT:     lwc1 $f21, 20($sp)
; LE-NEXT:     lwc1 $f22, 24($sp)
; LE-NEXT:     lwc1 $f23, 28($sp)
; BE-NEXT:     lwc1 $f21, 16($sp)
; BE-NEXT:     lwc1 $f20, 20($sp)
; BE-NEXT:     lwc1 $f23, 24($sp)
; BE-NEXT:     lwc1 $f22, 28($sp)
; ALL-NEXT:    jr $ra
  %x = fadd double %a, %b
  %p = alloca i8, i32 %n
  call void @h(ptr %p)
  %y = fmul double %x, %a
  ret double %y
}

; Spill and reload of a double around an inline asm that clobbers all FPRs.
define double @spill(double %a, double %b) {
; ALL-LABEL: spill:
; ALL:         mul.d $f0, $f0, $f0
; LE-NEXT:     swc1 $f0, 0($sp)
; LE-NEXT:     swc1 $f1, 4($sp)
; BE-NEXT:     swc1 $f1, 0($sp)
; BE-NEXT:     swc1 $f0, 4($sp)
; ALL-NEXT:    #APP
; ALL-NEXT:    #NO_APP
; LE-NEXT:     lwc1 $f0, 0($sp)
; LE-NEXT:     lwc1 $f1, 4($sp)
; BE-NEXT:     lwc1 $f1, 0($sp)
; BE-NEXT:     lwc1 $f0, 4($sp)
  %x = fadd double %a, %b
  call void asm sideeffect "", "~{$f0},~{$f1},~{$f2},~{$f3},~{$f4},~{$f5},~{$f6},~{$f7},~{$f8},~{$f9},~{$f10},~{$f11},~{$f12},~{$f13},~{$f14},~{$f15},~{$f16},~{$f17},~{$f18},~{$f19},~{$f20},~{$f21},~{$f22},~{$f23},~{$f24},~{$f25},~{$f26},~{$f27},~{$f28},~{$f29},~{$f30},~{$f31}"()
  %y = fmul double %x, %x
  ret double %y
}
