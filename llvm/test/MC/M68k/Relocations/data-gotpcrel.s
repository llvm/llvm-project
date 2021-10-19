; RUN: llvm-mc -triple m68k -filetype=obj %s -o - \
; RUN:   | llvm-readobj -r - | FileCheck -check-prefix=RELOC %s
; RUN: llvm-mc -triple m68k -show-encoding %s -o - \
; RUN:   | FileCheck -check-prefix=INSTR -check-prefix=FIXUP %s

; RELOC: R_68K_GOTPCREL8 dst1 0x1
; INSTR: move.l  (dst1@GOTPCREL,%pc,%d0), %a0
; FIXUP: fixup A - offset: 3, value: dst1@GOTPCREL+1, kind: FK_PCRel_1
move.l	(dst1@GOTPCREL,%pc,%d0), %a0

; RELOC: R_68K_GOTPCREL16 dst2 0x0
; INSTR: move.l  (dst2@GOTPCREL,%pc), %a0
; FIXUP: fixup A - offset: 2, value: dst2@GOTPCREL, kind: FK_PCRel_2
move.l	(dst2@GOTPCREL,%pc), %a0
