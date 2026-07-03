; RUN: llvm-mc -triple m68k -filetype=obj %s -o - \
; RUN:   | llvm-readobj -r - | FileCheck -check-prefix=RELOC %s
; RUN: llvm-mc -triple m68k -show-encoding %s -o - \
; RUN:   | FileCheck -check-prefix=INSTR %s

; RELOC: R_68K_GOTOFF8 dst1 0x0
; INSTR: move.l  (dst1@GOTOFF,%a5,%d0), %d0
move.l	(dst1@GOTOFF,%a5,%d0), %d0

; RELOC: R_68K_GOTOFF16 dst2 0x0
; INSTR: move.l  (dst2@GOTOFF,%a5), %d0
move.l	(dst2@GOTOFF,%a5), %d0

; RELOC: R_68K_GOTPCREL16 dst3 0x0
; INSTR: lea     (dst3@GOTPCREL,%pc), %a5
lea	(dst3@GOTPCREL,%pc), %a5
