; RUN: llvm-mc -triple m68k --mcpu=M68020 --position-independent -filetype=obj %s -o - \
; RUN:   | llvm-readobj -r - | FileCheck -check-prefix=RELOC %s
; RUN: llvm-mc -triple m68k --mcpu=M68020 --position-independent -show-encoding %s -o - \
; RUN:   | FileCheck -check-prefix=INSTR %s

; RELOC: R_68K_PLT16 target 0x0
; INSTR: jsr     (target@PLT,%pc)
jsr	(target@PLT,%pc)

; RELOC: R_68K_PLT32  __tls_get_addr 0x0
; INSTR: bsr.l   __tls_get_addr@PLT
bsr.l __tls_get_addr@PLT
