; RUN: llc < %s -march=avr | FileCheck --check-prefixes=CHECK,NOSECTIONS %s
; RUN: llc -function-sections -data-sections < %s -march=avr | FileCheck --check-prefixes=CHECK,SECTIONS %s

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

; NOSECTIONS: .section .rodata,"a",@progbits
; SECTIONS:   .section .rodata.ram1,"a",@progbits
; CHECK-LABEL: ram1:
@ram1 = constant i16 3

; NOSECTIONS: .data
; SECTIONS:   .section .data.ram2,"aw",@progbits
; CHECK-LABEL: ram2:
@ram2 = global i16 3
