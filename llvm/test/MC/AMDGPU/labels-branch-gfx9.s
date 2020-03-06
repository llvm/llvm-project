// RUN: llvm-mc -arch=amdgcn -mcpu=gfx900 -show-encoding %s | FileCheck %s --check-prefix=GFX9
// RUN: llvm-mc -arch=amdgcn -mcpu=gfx900 -filetype=obj %s | llvm-objdump -disassemble -mcpu=gfx900 - | FileCheck %s --check-prefix=BIN

loop_start:

s_call_b64 s[10:11], loop_end
// GFX9: s_call_b64 s[10:11], loop_end   ; encoding: [A,A,0x8a,0xba]
// GFX9-NEXT: ;   fixup A - offset: 0, value: loop_end, kind: fixup_si_sopp_br
// BIN: <loop_start>:
// BIN-NEXT: s_call_b64 s[10:11], loop_end // 000000000000: BA8A0001

s_call_b64 s[10:11], loop_start
// GFX9: s_call_b64 s[10:11], loop_start ; encoding: [A,A,0x8a,0xba]
// GFX9-NEXT: ;   fixup A - offset: 0, value: loop_start, kind: fixup_si_sopp_br
// BIN: s_call_b64 s[10:11], loop_start  // 000000000004: BA8AFFFE
// BIN: <loop_end>:

loop_end:
  s_nop 0
