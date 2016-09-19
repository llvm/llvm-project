// RUN: llvm-mc -arch=amdgcn -mcpu=fiji -show-encoding %s | FileCheck %s --check-prefix=VI
// RUN: llvm-mc -arch=amdgcn -mcpu=fiji -filetype=obj %s | llvm-objdump -disassemble -mcpu=fiji - | FileCheck %s --check-prefix=BIN

loop_start:
s_branch loop_start
// VI: s_branch loop_start ; encoding: [A,A,0x82,0xbf]
// VI-NEXT: ;   fixup A - offset: 0, value: loop_start, kind: fixup_si_sopp_br
// BIN: loop_start:
// BIN-NEXT: BF82FFFF

s_branch loop_end
// VI: s_branch loop_end ; encoding: [A,A,0x82,0xbf]
// VI-NEXT: ;   fixup A - offset: 0, value: loop_end, kind: fixup_si_sopp_br
// BIN: BF820000
// BIN: loop_end:
loop_end:

s_branch gds
// VI: s_branch gds ; encoding: [A,A,0x82,0xbf]
// VI-NEXT: ;   fixup A - offset: 0, value: gds, kind: fixup_si_sopp_br
// BIN: BF820000
// BIN: gds:
gds:
  s_nop 0
