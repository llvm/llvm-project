; RUN: llvm-mc -triple m68k -show-encoding %s | FileCheck -check-prefix=INSTR %s
; RUN: llvm-mc -triple m68k --mcpu=M68020 -filetype=obj %s -o %t
; RUN: llvm-readelf -rs %t | FileCheck --check-prefix=READELF %s --implicit-check-not=TLS

; INSTR:      move.l  (le@TPOFF,%a0), %d0
; INSTR-NEXT:                fixup A - offset: 2, value: le@TPOFF, kind: FK_Data_2
; INSTR-NEXT: move.l  (ie@GOTTPOFF,%a0), %a2
; INSTR-NEXT:                fixup A - offset: 2, value: ie@GOTTPOFF, kind: FK_Data_2
; INSTR-NEXT: lea     (gd@TLSGD,%a0), %a0
; INSTR-NEXT:                fixup A - offset: 2, value: gd@TLSGD, kind: FK_Data_2
; INSTR-NEXT: move.l  (ld@TLSLD,%a0), %d0
; INSTR-NEXT:                fixup A - offset: 2, value: ld@TLSLD, kind: FK_Data_2
; INSTR-NEXT: lea     (ld@TLSLDM,%a2), %a1
; INSTR-NEXT:                fixup A - offset: 2, value: ld@TLSLDM, kind: FK_Data_2

; READELF:      R_68K_TLS_LE16         00000000   le + 0
; READELF-NEXT: R_68K_TLS_IE16         00000000   ie + 0
; READELF-NEXT: R_68K_TLS_GD16         00000000   gd + 0
; READELF-NEXT: R_68K_TLS_LDO16        00000000   ld + 0
; READELF-NEXT: R_68K_TLS_LDM16        00000000   ld + 0

; READELF:      TLS GLOBAL DEFAULT UND le
; READELF-NEXT: TLS GLOBAL DEFAULT UND ie
; READELF-NEXT: TLS GLOBAL DEFAULT UND gd
; READELF-NEXT: TLS GLOBAL DEFAULT UND ld

move.l  (le@TPOFF,%a0), %d0
move.l  (ie@GOTTPOFF,%a0), %a2
lea     (gd@TLSGD,%a0), %a0
move.l  (ld@TLSLD,%a0), %d0
lea     (ld@TLSLDM,%a2), %a1
