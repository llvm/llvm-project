; RUN: llc -mtriple=mips -mcpu=mips32r2 %s -o %t.32.s
; RUN: FileCheck %s < %t.32.s
; RUN: llc -mtriple=mips -mcpu=mips32r2 -filetype=obj %s -o %t.32.o
; RUN: llvm-mc -triple=mips -mcpu=mips32r2 -filetype=obj %t.32.s -o %t.32.reassembled.o
; RUN: llvm-objcopy --dump-section .text=%t.32.text %t.32.o
; RUN: llvm-objcopy --dump-section .text=%t.32.reassembled.text %t.32.reassembled.o
; RUN: cmp %t.32.text %t.32.reassembled.text
; RUN: llc -mtriple=mips64el -mcpu=mips64r2 %s -o %t.64.s
; RUN: FileCheck %s < %t.64.s
; RUN: llc -mtriple=mips64el -mcpu=mips64r2 -filetype=obj %s -o %t.64.o
; RUN: llvm-mc -triple=mips64el -mcpu=mips64r2 -filetype=obj %t.64.s -o %t.64.reassembled.o
; RUN: llvm-objcopy --dump-section .text=%t.64.text %t.64.o
; RUN: llvm-objcopy --dump-section .text=%t.64.reassembled.text %t.64.reassembled.o
; RUN: cmp %t.64.text %t.64.reassembled.text
; RUN: llc -mtriple=mips -mcpu=mips32r2 -mattr=+micromips %s -o - \
; RUN:   | FileCheck %s --check-prefix=MICROMIPS

; The integrated assembler has already expanded these macros and filled their
; branch delay slots. Reassembling its output must not fill the slots again.
define i32 @div(i32 %a, i32 %b) {
; CHECK-LABEL: div:
; CHECK:       #APP
; CHECK-NEXT:  .set push
; CHECK-NEXT:  .set at
; CHECK-NEXT:  .set macro
; CHECK-NEXT:  .set reorder
; CHECK:       div $zero, $[[A:[0-9]+]], $[[B:[0-9]+]]
; CHECK-NEXT:  .set push
; CHECK-NEXT:  .set noreorder
; CHECK-NEXT:  bnez $[[B]], {{.*}}
; CHECK-NEXT:  nop
; CHECK-NEXT:  .set pop
; CHECK-NEXT:  break 7
; CHECK:       mflo $[[A]]
; CHECK-NEXT:  mflo $2
; CHECK:       .set pop
; CHECK-NEXT:  #NO_APP
  %result = call i32 asm sideeffect "div $1, $2\0A\09mflo $0", "=r,r,r,~{hi},~{lo},~{$1}"(i32 %a, i32 %b)
  ret i32 %result
}

; The following functions are only checked by comparing the reassembled code.
define i32 @remu(i32 %a, i32 %b) {
  %result = call i32 asm sideeffect "remu $0, $1, $2", "=r,r,r,~{hi},~{lo},~{$1}"(i32 %a, i32 %b)
  ret i32 %result
}

; Keep the parser's default reorder mode: user branches still need delay slots.
define void @branch() {
; CHECK-LABEL: branch:
; CHECK:       #APP
; CHECK:       .set noreorder
; CHECK:       b {{.*}}
; CHECK-NEXT:  nop
; CHECK-NEXT:  .set reorder
; CHECK:       .set pop
; CHECK-NEXT:  #NO_APP
  call void asm sideeffect "b 1f\0A1:", "~{$1}"()
  ret void
}

; Explicitly supplied delay slots must also survive reassembly unchanged.
define void @branch_noreorder() {
; CHECK-LABEL: branch_noreorder:
; CHECK:       #APP
; CHECK:       b {{.*}}
; CHECK-NEXT:  nop
; CHECK-NEXT:  {{.*}}:
; CHECK:       .set pop
; CHECK-NEXT:  #NO_APP
  call void asm sideeffect ".set noreorder\0Ab 1f\0Anop\0A1:", "~{$1}"()
  ret void
}

; A preceding branch restores reorder in the emitted assembly.
define i32 @branch_div(i32 %a, i32 %b) {
; CHECK-LABEL: branch_div:
; CHECK:       #APP
; CHECK:       b {{.*}}
; CHECK-NEXT:  nop
; CHECK-NEXT:  .set reorder
; CHECK:       div $zero,
; CHECK-NEXT:  .set push
; CHECK-NEXT:  .set noreorder
; CHECK-NEXT:  bnez
; CHECK-NEXT:  nop
; CHECK-NEXT:  .set pop
; CHECK:       mflo
; CHECK:       b {{.*}}
; CHECK-NEXT:  nop
; CHECK-NEXT:  .set reorder
  %result = call i32 asm sideeffect "b 1f\0A1:\0Adiv $0, $1, $2\0Ab 2f\0A2:", "=r,r,r,~{hi},~{lo},~{$1}"(i32 %a, i32 %b)
  ret i32 %result
}

; Explicit reorder must not make expanded macro delay slots grow.
define i32 @reorder_rem(i32 %a, i32 %b) {
; CHECK-LABEL: reorder_rem:
; CHECK:       #APP
; CHECK:       div $zero,
; CHECK-NEXT:  .set push
; CHECK-NEXT:  .set noreorder
; CHECK-NEXT:  bnez
; CHECK-NEXT:  nop
; CHECK-NEXT:  .set pop
; CHECK:       mfhi
; CHECK:       b {{.*}}
; CHECK-NEXT:  nop
; CHECK-NEXT:  .set reorder
  %result = call i32 asm sideeffect ".set reorder\0Arem $0, $1, $2\0Ab 1f\0A1:", "=r,r,r,~{hi},~{lo},~{$1}"(i32 %a, i32 %b)
  ret i32 %result
}

; In noreorder mode the expansion is emitted as is.
define i32 @noreorder_divu(i32 %a, i32 %b) {
; CHECK-LABEL: noreorder_divu:
; CHECK:       #APP
; CHECK:       divu $zero,
; CHECK-NEXT:  bnez
; CHECK-NEXT:  nop
; CHECK-NEXT:  break 7
; CHECK:       mflo
; CHECK:       b {{.*}}
; CHECK-NEXT:  addiu $3, $3, 1
; CHECK-NEXT:  {{.*}}:
  %result = call i32 asm sideeffect ".set noreorder\0Adivu $0, $1, $2\0Ab 1f\0Aaddiu $$3, $$3, 1\0A1:", "=r,r,r,~{hi},~{lo},~{$1},~{$3}"(i32 %a, i32 %b)
  ret i32 %result
}

define i32 @mulo(i32 %a, i32 %b) {
  %result = call i32 asm sideeffect "mulo $0, $1, $2", "=r,r,r,~{hi},~{lo},~{$1}"(i32 %a, i32 %b)
  ret i32 %result
}

define i32 @mulou(i32 %a, i32 %b) {
  %result = call i32 asm sideeffect "mulou $0, $1, $2", "=r,r,r,~{hi},~{lo},~{$1}"(i32 %a, i32 %b)
  ret i32 %result
}

define i32 @abs(i32 %a) {
  %result = call i32 asm sideeffect "abs $0, $0", "=r,0,~{$1}"(i32 %a)
  ret i32 %result
}

define void @jal(ptr %f) {
  call void asm sideeffect "jal $0", "r,~{$1},~{$31}"(ptr %f)
  ret void
}

; The microMIPS b pseudo fills its own delay slot too. Only check the text, as
; its 16-bit nop is printed as a plain nop.
define void @b_micromips() {
; MICROMIPS-LABEL: b_micromips:
; MICROMIPS:       .set reorder
; MICROMIPS:       .set push
; MICROMIPS-NEXT:  .set noreorder
; MICROMIPS-NEXT:  b16 16
; MICROMIPS-NEXT:  nop
; MICROMIPS-NEXT:  .set pop
  call void asm sideeffect "b 16", ""()
  ret void
}
