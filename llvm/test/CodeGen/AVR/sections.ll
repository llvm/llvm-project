; RUN: llc < %s -mtriple=avr --mcpu=atxmega384d3 \
; RUN:     | FileCheck --check-prefixes=CHECK,NOSECTIONS %s
; RUN: llc -function-sections -data-sections < %s -mtriple=avr --mcpu=atxmega384d3 \
; RUN:     | FileCheck --check-prefixes=CHECK,SECTIONS %s
; RUN: not llc -function-sections -data-sections < %s -mtriple=avr --mcpu=at90s8515 2>&1 \
; RUN:     | FileCheck --check-prefixes=CHECK-8515 %s
; RUN: not llc -function-sections -data-sections < %s -mtriple=avr --mcpu=attiny40 2>&1 \
; RUN:     | FileCheck --check-prefixes=CHECK-tiny40 %s

; Test that functions (in address space 1) are not considered .progmem data.

; CHECK: .text
; SECTIONS: .text.somefunc,"ax",@progbits
; CHECK-LABEL: somefunc:
define void @somefunc() addrspace(1) {
  ret void
}


; Test whether global variables are placed in the correct section.

; Note: avr-gcc would place this global in .progmem.data.flash with
; -fdata-sections. The AVR backend does not yet respect -fdata-sections in this
; case.

; CHECK: .section .progmem.data,"a",@progbits
; CHECK-LABEL: flash:
@flash = addrspace(1) constant i16 3

; CHECK: .section .progmem1.data,"a",@progbits
; CHECK-LABEL: flash1:
; CHECK-8515: error: Current AVR subtarget does not support accessing extended program memory
; CHECK-tiny40: error: Current AVR subtarget does not support accessing program memory
@flash1 = addrspace(2) constant i16 4

; CHECK: .section .progmem2.data,"a",@progbits
; CHECK-LABEL: flash2:
; CHECK-8515: error: Current AVR subtarget does not support accessing extended program memory
; CHECK-tiny40: error: Current AVR subtarget does not support accessing program memory
@flash2 = addrspace(3) constant i16 5

; CHECK: .section .progmem3.data,"a",@progbits
; CHECK-LABEL: flash3:
; CHECK-8515: error: Current AVR subtarget does not support accessing extended program memory
; CHECK-tiny40: error: Current AVR subtarget does not support accessing program memory
@flash3 = addrspace(4) constant i16 6

; CHECK: .section .progmem4.data,"a",@progbits
; CHECK-LABEL: flash4:
; CHECK-8515: error: Current AVR subtarget does not support accessing extended program memory
; CHECK-tiny40: error: Current AVR subtarget does not support accessing program memory
@flash4 = addrspace(5) constant i16 7

; CHECK: .section .progmem5.data,"a",@progbits
; CHECK-LABEL: flash5:
; CHECK-8515: error: Current AVR subtarget does not support accessing extended program memory
; CHECK-tiny40: error: Current AVR subtarget does not support accessing program memory
@flash5 = addrspace(6) constant i16 8

; NOSECTIONS: .section .rodata,"a",@progbits
; SECTIONS:   .section .rodata.ram1,"a",@progbits
; CHECK-LABEL: ram1:
@ram1 = constant i16 3

; NOSECTIONS: .data
; SECTIONS:   .section .data.ram2,"aw",@progbits
; CHECK-LABEL: ram2:
@ram2 = global i16 3
