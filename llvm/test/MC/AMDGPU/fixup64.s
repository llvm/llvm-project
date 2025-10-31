// RUN: not llvm-mc -triple=amdgcn -mcpu=tahiti -show-encoding %s | FileCheck --check-prefix=SI %s
// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s | FileCheck --check-prefix=GFX1250 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=tahiti -show-encoding %s 2>&1 | FileCheck --check-prefix=SI-ERR --implicit-check-not=error: --strict-whitespace %s

.LL1:
.LL2:
s_mov_b64 vcc, .LL2-.LL1
// GFX1250:         s_mov_b64 vcc, .LL2-.LL1        ; encoding: [0xfe,0x01,0xea,0xbe,A,A,A,A,A,A,A,A]
// GFX1250-NEXT:                                    ;   fixup A - offset: 4, value: .LL2-.LL1, kind: FK_Data_8
// SI-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: invalid operand for instruction

s_mov_b32 s0, .LL2-.LL1
// SI: s_mov_b32 s0, .LL2-.LL1                      ; encoding: [0xff,0x03,0x80,0xbe,A,A,A,A]
// SI-NEXT:                                         ;   fixup A - offset: 4, value: .LL2-.LL1, kind: FK_Data_4

// GFX1250: s_mov_b32 s0, .LL2-.LL1                 ; encoding: [0xff,0x00,0x80,0xbe,A,A,A,A]
// GFX1250-NEXT:                                    ;   fixup A - offset: 4, value: .LL2-.LL1, kind: FK_Data_4

s_mov_b64 s[0:1], sym@abs64
// GFX1250:        s_mov_b64 s[0:1], sym@abs64      ; encoding: [0xfe,0x01,0x80,0xbe,A,A,A,A,A,A,A,A]
// GFX1250-NEXT:                                    ;   fixup A - offset: 4, value: sym@abs64, kind: FK_Data_8
// SI-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: invalid operand for instruction

s_mov_b64 s[0:1], callee@rel64
// GFX1250: s_mov_b64 s[0:1], callee@rel64          ; encoding: [0xfe,0x01,0x80,0xbe,A,A,A,A,A,A,A,A]
// GFX1250-NEXT:                                    ;   fixup A - offset: 4, value: callee@rel64, kind: FK_PCRel_8
// SI-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: invalid operand for instruction
