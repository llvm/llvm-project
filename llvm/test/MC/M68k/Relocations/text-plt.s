; RUN: llvm-mc -triple m68k -filetype=obj %s -o - \
; RUN:   | llvm-readobj -r - | FileCheck -check-prefix=RELOC %s
; RUN: llvm-mc -triple m68k -show-encoding %s -o - \
; RUN:   | FileCheck -check-prefix=INSTR -check-prefix=FIXUP %s

; RELOC: R_68K_PLT16 target 0x0
; INSTR: jsr     (target@PLT,%pc)
; FIXUP: fixup A - offset: 2, value: target@PLT, kind: FK_PCRel_2
jsr	(target@PLT,%pc)
