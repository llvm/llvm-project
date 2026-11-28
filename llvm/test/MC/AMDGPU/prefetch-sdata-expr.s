# RUN: split-file --leading-lines %s %t
# RUN: llvm-mc -triple=amdgpu12.00 -show-encoding %t/valid.s | FileCheck %s --check-prefix=ENC
# RUN: llvm-mc -triple=amdgpu12.00 -filetype=obj %t/valid.s | llvm-objdump -d --mcpu=gfx1200 - | FileCheck %s --check-prefix=BIN
# RUN: not llvm-mc -triple=amdgpu12.00 -filetype=obj -o /dev/null %t/range.s 2>&1 | FileCheck %s --check-prefix=RANGE
# RUN: not llvm-mc -triple=amdgpu12.00 -filetype=obj -o /dev/null %t/undef.s 2>&1 | FileCheck %s --check-prefix=UNDEF
# RUN: not llvm-mc -triple=amdgpu12.00 -filetype=obj -o /dev/null %t/specifier.s 2>&1 | FileCheck %s --check-prefix=SPECIFIER
# RUN: not llvm-mc -triple=amdgpu12.00 -filetype=obj -o /dev/null %t/scratch.s 2>&1 | FileCheck %s --check-prefix=SCRATCH

# A layout-dependent sdata expression is resolved by fixup_si_prefetch_sdata.

#--- valid.s
begin:
s_prefetch_inst_pc_rel begin, null, (end - begin) / 8
# ENC: s_prefetch_inst_pc_rel begin, null, (end-begin)/8 ; encoding: [0bBB000000,0b10100BBB,0x04,0xf4,A,A,A,0xf8]
# ENC-NEXT: ;   fixup A - offset: 4, value: begin, kind: fixup_si_smem_pcrel
# ENC-NEXT: ;   fixup B - offset: 0, value: (end-begin)/8, kind: fixup_si_prefetch_sdata
# BIN: s_prefetch_inst_pc_rel begin, null, 2 // 000000000000: F404A080 F8FFFFF8
s_nop 0
s_nop 0
end:

#--- range.s
# RANGE: range.s:[[#@LINE+1]]:{{[0-9]+}}: error: prefetch sdata exceeds 5-bit range
s_prefetch_inst_pc_rel begin, null, end - begin
begin:
.space 32
end:

#--- undef.s
# UNDEF: undef.s:[[#@LINE+1]]:{{[0-9]+}}: error: expected an absolute expression
s_prefetch_inst_pc_rel begin, null, undef_sym
begin:

#--- specifier.s
# A relocation specifier must not select a relocation that covers the
# whole instruction word.
# SPECIFIER: specifier.s:[[#@LINE+1]]:{{[0-9]+}}: error: expected an absolute expression
s_prefetch_inst_pc_rel 0, null, external@abs32@lo

#--- scratch.s
# SCRATCH: scratch.s:[[#@LINE+1]]:{{[0-9]+}}: error: expected an absolute expression
s_prefetch_inst_pc_rel 0, null, SCRATCH_RSRC_DWORD0
