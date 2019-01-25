; RUN: llc < %s | FileCheck %s

; Check that absolute addressing mode is represented in a way
; defined in MSP430 EABI and not as indexed addressing mode form.
; See PR39993 for details.

target datalayout = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8-n8:16"
target triple = "msp430-elf"

define void @f() {
entry:
; CHECK: mov r1, &256
  call void asm sideeffect "mov r1, $0", "*m"(i8* inttoptr (i16 256 to i8*))
  ret void
}
