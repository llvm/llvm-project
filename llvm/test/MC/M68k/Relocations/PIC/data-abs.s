; RUN: llvm-mc -triple m68k -filetype=obj --position-independent %s -o - \
; RUN:   | llvm-readobj -r - | FileCheck -check-prefix=RELOC %s
; RUN: llvm-mc -triple m68k -show-encoding --position-independent %s -o - \
; RUN:   | FileCheck -check-prefix=INSTR -check-prefix=FIXUP %s

; RELOC: R_68K_32 dst 0x0
; INSTR: move.l dst, %d0
; FIXUP: fixup A - offset: 2, value: dst, kind: FK_Data_4
move.l	dst, %d0

; Relocating immediate values

; RELOC: R_68K_8 str8 0x0
; INSTR: move.b  #str8,  (4,%sp)
; FIXUP: fixup A - offset: 3, value: str8, kind: FK_Data_1
move.b  #str8,  (4,%sp)

; RELOC: R_68K_16 str16 0x0
; INSTR: move.w  #str16,  (4,%sp)
; FIXUP: fixup A - offset: 2, value: str16, kind: FK_Data_2
move.w  #str16, (4,%sp)

; RELOC: R_68K_32 str32 0x0
; INSTR: move.l  #str32,  (4,%sp)
; FIXUP: fixup A - offset: 2, value: str32, kind: FK_Data_4
move.l  #str32, (4,%sp)
