; RUN: llc < %s -mtriple=avr -mcpu=attiny85 -filetype=obj -o - | llvm-objdump --mcpu=attiny85 -dr --no-show-raw-insn --no-leading-addr - | FileCheck --check-prefix=ATTINY85 %s
; RUN: llc < %s -mtriple=avr -mcpu=avr3 -filetype=obj -o - | llvm-objdump --mcpu=avr3 -dr --no-show-raw-insn --no-leading-addr - | FileCheck --check-prefix=AVR3 %s

; ATTINY85: <main>:
; ATTINY85-NEXT: andi r24, 0x1
; ATTINY85: cpi r24, 0x0
; ATTINY85-NEXT: breq .-2
; ATTINY85-NEXT: R_AVR_7_PCREL .text+0x100c
; ATTINY85-NEXT: rjmp .-2
; ATTINY85-NEXT: R_AVR_13_PCREL .text+0x2
; ATTINY85: ldi r24, 0x3
; ATTINY85-NEXT: ret

; AVR3: <main>:
; AVR3-NEXT: andi r24, 0x1
; AVR3: cpi r24, 0x0
; AVR3-NEXT: breq .-2
; AVR3-NEXT: R_AVR_7_PCREL .text+0x100e
; AVR3-NEXT: jmp 0x0
; AVR3-NEXT: R_AVR_CALL .text+0x2
; AVR3: ldi r24, 0x3
; AVR3-NEXT: ret

define i8 @main(i1 %a) {
entry-block:
  br label %hello
hello:
  call void asm sideeffect ".space 4100", ""()
  br i1 %a, label %hello, label %finished
finished:
  ret i8 3
}
