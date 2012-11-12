; RUN: llc %s -mtriple=thumbv5-linux-gnueabi -mcpu=xscale -o - | \
; RUN: FileCheck -check-prefix=ASM %s

; RUN: llc %s -mtriple=thumbv5-linux-gnueabi -filetype=obj \
; RUN: -mcpu=xscale -o - | elf-dump --dump-section-data | \
; RUN: FileCheck -check-prefix=OBJ %s

; FIXME: The OBJ test should be a .s to .o test and the ASM test should
; be moved to test/CodeGen/ARM.

define void @foo() nounwind {
entry:
  ret void
}

; ASM:           .eabi_attribute 6, 5
; ASM-NEXT:      .eabi_attribute 8, 1
; ASM-NEXT:      .eabi_attribute 9, 1

; OBJ:           Section 4
; OBJ-NEXT:      'sh_name', 0x0000000c
; OBJ-NEXT:      'sh_type', 0x70000003
; OBJ-NEXT:	   'sh_flags', 0x00000000
; OBJ-NEXT:	   'sh_addr', 0x00000000
; OBJ-NEXT:	   'sh_offset', 0x00000038
; OBJ-NEXT:	   'sh_size', 0x00000020
; OBJ-NEXT:	   'sh_link', 0x00000000
; OBJ-NEXT:	   'sh_info', 0x00000000
; OBJ-NEXT:	   'sh_addralign', 0x00000001
; OBJ-NEXT:	   'sh_entsize', 0x00000000
; OBJ-NEXT:      '_section_data', '411f0000 00616561 62690001 15000000 06050801 09011401 15011703 18011901'
