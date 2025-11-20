; RUN: llvm-mc -triple m68k -filetype=obj --position-independent %s -o - \
; RUN:   | llvm-readobj -r - | FileCheck -check-prefix=RELOC %s
; RUN: llvm-mc -triple m68k -show-encoding --position-independent %s -o - \
; RUN:   | FileCheck -check-prefix=INSTR %s

; RELOC: R_68K_32 dst 0x0
; INSTR: move.l dst, %d0
move.l	dst, %d0

; Relocating immediate values

; RELOC: R_68K_8 str8 0x0
; INSTR: move.b  #str8,  (4,%sp)
move.b  #str8,  (4,%sp)

; RELOC: R_68K_16 str16 0x0
; INSTR: move.w  #str16,  (4,%sp)
move.w  #str16, (4,%sp)

; RELOC: R_68K_32 str32 0x0
; INSTR: move.l  #str32,  (4,%sp)
move.l  #str32, (4,%sp)
