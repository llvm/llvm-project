; RUN: llvm-mc -triple m68k -filetype=obj %s -o - \
; RUN:   | llvm-readobj -r - | FileCheck -check-prefix=RELOC %s
; RUN: llvm-mc -triple m68k -show-encoding %s -o - \
; RUN:   | FileCheck -check-prefix=INSTR -check-prefix=FIXUP %s

; RELOC: R_68K_32 dst 0x0
; INSTR: move.l dst, %d0
; FIXUP: fixup A - offset: 2, value: dst, kind: FK_Data_4
move.l	dst, %d0
