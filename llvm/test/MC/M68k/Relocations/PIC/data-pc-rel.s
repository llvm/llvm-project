; RUN: llvm-mc -triple m68k -filetype=obj --position-independent %s -o - \
; RUN:   | llvm-readobj -r - | FileCheck -check-prefix=RELOC %s
; RUN: llvm-mc -triple m68k -show-encoding --position-independent %s -o - \
; RUN:   | FileCheck -check-prefix=INSTR -check-prefix=FIXUP %s

; RELOC: R_68K_PC8 dst1 0x1
; INSTR: move.l  (dst1,%pc,%a0), %a0
; FIXUP: fixup A - offset: 3, value: dst1+1, kind: FK_PCRel_1
move.l	(dst1,%pc,%a0), %a0

; RELOC: R_68K_PC16 dst2 0x0
; INSTR: move.l  (dst2,%pc), %a0
; FIXUP: fixup A - offset: 2, value: dst2, kind: FK_PCRel_2
move.l	(dst2,%pc), %a0

; Shouldn't have any relocation
; RELOC-NOT: R_68K_PC
; INSTR: move.l  (0,%pc), %a0
; FIXUP-NOT: fixup
move.l	(0,%pc), %a0
