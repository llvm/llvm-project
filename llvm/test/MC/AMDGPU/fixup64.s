// RUN: llvm-mc -triple=amdgcn -mcpu=tahiti -show-encoding %s | FileCheck --check-prefix=SI %s
// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1210 -show-encoding %s | FileCheck --check-prefix=GFX1210 %s

// Make sure it does not crash on SI

.LL1:
.LL2:
s_mov_b64 vcc, .LL2-.LL1
// SI:         s_mov_b64 vcc, .LL2-.LL1             ; encoding: [0xff,0x04,0xea,0xbe,A,A,A,A]
// SI-NEXT:                                         ;   fixup A - offset: 4, value: .LL2-.LL1, kind: FK_Data_4

// GFX1210:         s_mov_b64 vcc, .LL2-.LL1        ; encoding: [0xfe,0x01,0xea,0xbe,A,A,A,A,A,A,A,A]
// GFX1210-NEXT:                                    ;   fixup A - offset: 4, value: .LL2-.LL1, kind: FK_Data_8


s_mov_b32 s0, .LL2-.LL1
// SI: s_mov_b32 s0, .LL2-.LL1                      ; encoding: [0xff,0x03,0x80,0xbe,A,A,A,A]
// SI-NEXT:                                         ;   fixup A - offset: 4, value: .LL2-.LL1, kind: FK_Data_4

// GFX1210: s_mov_b32 s0, .LL2-.LL1                 ; encoding: [0xff,0x00,0x80,0xbe,A,A,A,A]
// GFX1210-NEXT:                                    ;   fixup A - offset: 4, value: .LL2-.LL1, kind: FK_Data_4

s_mov_b64 s[0:1], sym@abs64
// SI: s_mov_b64 s[0:1], sym@abs64                  ; encoding: [0xff,0x04,0x80,0xbe,A,A,A,A]
// SI-NEXT:                                         ;   fixup A - offset: 4, value: sym@abs64, kind: FK_Data_4

// GFX1210:        s_mov_b64 s[0:1], sym@abs64      ; encoding: [0xfe,0x01,0x80,0xbe,A,A,A,A,A,A,A,A]
// GFX1210-NEXT:                                    ;   fixup A - offset: 4, value: sym@abs64, kind: FK_Data_8
