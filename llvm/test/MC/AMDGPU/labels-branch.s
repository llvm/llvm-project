// RUN: llvm-mc -triple=amdgpu8.03 -show-encoding %s | FileCheck %s --check-prefix=VI
// RUN: llvm-mc -triple=amdgpu8.03 -filetype=obj %s | llvm-objdump -d --mcpu=fiji - | FileCheck %s --check-prefix=BIN
// RUN: llvm-mc -triple=amdgpu12.00 -defsym=GFX12=1 -show-encoding %s | FileCheck %s --check-prefix=GFX12
// RUN: llvm-mc -triple=amdgpu12.00 -defsym=GFX12=1 -filetype=obj %s | llvm-objdump -d --mcpu=gfx1200 - | FileCheck %s --check-prefix=GFX12-BIN
// RUN: llvm-mc -triple=amdgpu13.10 -defsym=GFX13=1 -show-encoding %s | FileCheck %s --check-prefix=GFX13
// RUN: llvm-mc -triple=amdgpu13.10 -defsym=GFX13=1 -filetype=obj %s | llvm-objdump -d --mcpu=gfx1310 - | FileCheck %s --check-prefix=GFX13-BIN

loop_start:
s_branch loop_start
// VI: s_branch loop_start ; encoding: [A,A,0x82,0xbf]
// VI-NEXT: ;   fixup A - offset: 0, value: loop_start, kind: fixup_si_sopp_br
// BIN: <loop_start>:
// BIN-NEXT: s_branch loop_start // 000000000000: BF82FFFF

s_branch loop_end
// VI: s_branch loop_end ; encoding: [A,A,0x82,0xbf]
// VI-NEXT: ;   fixup A - offset: 0, value: loop_end, kind: fixup_si_sopp_br
// BIN: s_branch loop_end // 000000000004: BF820000
// BIN: <loop_end>:
loop_end:

s_branch gds
// VI: s_branch gds ; encoding: [A,A,0x82,0xbf]
// VI-NEXT: ;   fixup A - offset: 0, value: gds, kind: fixup_si_sopp_br
// BIN: s_branch gds // 000000000008: BF820000
// BIN: <gds>:
gds:
  s_nop 0

// s_prefetch_inst_pc_rel encodes a signed 24-bit byte offset from the next
// instruction (PC + 8). GFX8 has no such instruction.
.ifdef GFX12
.section .text.prefetch,"ax",@progbits
pref_back:
s_prefetch_inst_pc_rel pref_back, s10, 7
// GFX12: s_prefetch_inst_pc_rel pref_back, s10, 7 ; encoding: [0xc0,0xa1,0x04,0xf4,A,A,A,0x14]
// GFX12-NEXT: ;   fixup A - offset: 4, value: pref_back, kind: fixup_si_smem_pcrel
// GFX12-BIN: <pref_back>:
// GFX12-BIN-NEXT: s_prefetch_inst_pc_rel pref_back, s10, 7 // 000000000000: F404A1C0 14FFFFF8

s_prefetch_inst_pc_rel pref_zero, s10, 7
// GFX12: s_prefetch_inst_pc_rel pref_zero, s10, 7 ; encoding: [0xc0,0xa1,0x04,0xf4,A,A,A,0x14]
// GFX12-NEXT: ;   fixup A - offset: 4, value: pref_zero, kind: fixup_si_smem_pcrel
// GFX12-BIN-NEXT: s_prefetch_inst_pc_rel pref_zero, s10, 7 // 000000000008: F404A1C0 14000000

pref_zero:
s_prefetch_inst_pc_rel pref_plus4, s10, 7
// GFX12: s_prefetch_inst_pc_rel pref_plus4, s10, 7 ; encoding: [0xc0,0xa1,0x04,0xf4,A,A,A,0x14]
// GFX12-NEXT: ;   fixup A - offset: 4, value: pref_plus4, kind: fixup_si_smem_pcrel
// GFX12-BIN: <pref_zero>:
// GFX12-BIN-NEXT: s_prefetch_inst_pc_rel pref_plus4, s10, 7 // 000000000010: F404A1C0 14000004
s_nop 0
pref_plus4:
s_nop 0
// GFX12-BIN: <pref_plus4>:
.endif

// The displacement field is the same on GFX13. The opcode is 0x23, so the
// resolved first word is F40461C0 rather than GFX12's F404A1C0.
.ifdef GFX13
.section .text.prefetch13,"ax",@progbits
pref13_back:
s_prefetch_inst_pc_rel pref13_back, s10, 7
// GFX13: s_prefetch_inst_pc_rel pref13_back, s10, 7 ; encoding: [0xc0,0x61,0x04,0xf4,A,A,A,0x14]
// GFX13-NEXT: ;   fixup A - offset: 4, value: pref13_back, kind: fixup_si_smem_pcrel
// GFX13-BIN: <pref13_back>:
// GFX13-BIN-NEXT: s_prefetch_inst_pc_rel pref13_back, s10, 7 // 000000000000: F40461C0 14FFFFF8
.endif
