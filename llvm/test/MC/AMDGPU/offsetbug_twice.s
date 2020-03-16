// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1010 -show-encoding %s | FileCheck %s --check-prefix=GFX10
// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1010 -filetype=obj %s | llvm-objdump -d --mcpu=gfx1010 - | FileCheck %s --check-prefix=BIN
	s_getpc_b64 s[0:1]
	s_cbranch_vccnz BB0_2
// GFX10: s_cbranch_vccnz BB0_2           ; encoding: [A,A,0x87,0xbf]
// GFX10-NEXT: ;   fixup A - offset: 0, value: BB0_2, kind: fixup_si_sopp_br
// BIN: s_cbranch_vccnz BB0_2 // 000000000004: BF870061
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	v_nop
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_cbranch_vccnz BB0_1
// GFX10: s_cbranch_vccnz BB0_1           ; encoding: [A,A,0x87,0xbf]
// GFX10-NEXT: ;   fixup A - offset: 0, value: BB0_1, kind: fixup_si_sopp_br
// BIN: s_cbranch_vccnz BB0_1 // 000000000064: BF870041
	s_nop 0
	s_cbranch_execz BB0_3
// GFX10: s_cbranch_execz BB0_3           ; encoding: [A,A,0x88,0xbf]
// GFX10-NEXT: ;   fixup A - offset: 0, value: BB0_3, kind: fixup_si_sopp_br
// BIN: s_cbranch_execz BB0_3 // 00000000006C: BF880040
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
BB0_1:
	s_nop 0
BB0_3:
	s_nop 0
	s_nop 0
	s_nop 0
	s_cbranch_vccnz BB0_2
// GFX10: s_cbranch_vccnz BB0_2           ; encoding: [A,A,0x87,0xbf]
// GFX10-NEXT: ;   fixup A - offset: 0, value: BB0_2, kind: fixup_si_sopp_br
// BIN: s_cbranch_vccnz BB0_2 // 00000000017C: BF870003
	s_nop 0
	s_nop 0
	s_nop 0
BB0_2:
	s_nop 0
	s_nop 0
	s_endpgm
