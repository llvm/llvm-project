; RUN: llvm-mc -triple m68k -filetype=obj --position-independent %s -o - \
; RUN:   | llvm-readobj -r - | FileCheck -check-prefix=RELOC %s
; RUN: llvm-mc -triple m68k -show-encoding --position-independent %s -o - \
; RUN:   | FileCheck -check-prefix=INSTR %s

; RELOC: R_68K_GOTPCREL8 dst1 0x1
; INSTR: move.l  (dst1@GOTPCREL,%pc,%d0), %a0
move.l	(dst1@GOTPCREL,%pc,%d0), %a0

; RELOC: R_68K_GOTPCREL16 dst2 0x0
; INSTR: move.l  (dst2@GOTPCREL,%pc), %a0
move.l	(dst2@GOTPCREL,%pc), %a0
