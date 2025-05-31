; RUN: llvm-mc -triple m68k --mcpu=M68020 -filetype=obj %s -o - \
; RUN:   | llvm-readobj -r - | FileCheck -check-prefix=RELOC %s
; RUN: llvm-mc -triple m68k --mcpu=M68020 -show-encoding %s -o - \
; RUN:   | FileCheck -check-prefix=INSTR -check-prefix=FIXUP %s

; RELOC: R_68K_PLT16 target 0x0
; INSTR: jsr     (target@PLT,%pc)
; FIXUP: fixup A - offset: 2, value: target@PLT, kind: FK_PCRel_2
jsr	(target@PLT,%pc)

; RELOC: R_68K_PLT32  __tls_get_addr 0x0
; INSTR: bsr.l   __tls_get_addr@PLT
; FIXUP: fixup A - offset: 2, value: __tls_get_addr@PLT, kind: FK_PCRel_4
bsr.l __tls_get_addr@PLT
