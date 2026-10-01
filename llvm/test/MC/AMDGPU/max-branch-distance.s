// RUN: not llvm-mc -triple=amdgpu6.00 -filetype=obj -o /dev/null %s 2>&1 | FileCheck -check-prefix=ERROR %s
// RUN: not llvm-mc -triple=amdgpu12.50 -filetype=obj -o /dev/null %s 2>&1 | FileCheck -check-prefix=ERROR %s

// On subtargets with FeatureUseAddPC64Inst, out of range branches are silently
// relaxed into s_add_pc_i64 instead of being diagnosed.
// RUN: llvm-mc -triple=amdgpu12.51 -filetype=obj -o %t.o %s 2>&1 | count 0
// RUN: llvm-objdump -d --disassemble-symbols=branch,cbranch,end %t.o | FileCheck -check-prefix=RELAX %s

// fill v_nop
LBB0_0:
    .fill 32768, 4, 0x0000007e

// ERROR: max-branch-distance.s:[[@LINE+5]]:12: error: branch size exceeds simm16
// The unconditional branch grows from 4 to 8 bytes.
// RELAX:      [[#%.16x,BR:]] <branch>:
// RELAX-NEXT: s_add_pc_i64
branch:
  s_branch LBB0_0

// ERROR: max-branch-distance.s:[[@LINE+6]]:18: error: branch size exceeds simm16
// The conditional branch grows from 4 to 12 bytes.
// RELAX:      [[#%.16x,BR+8]] <cbranch>:
// RELAX-NEXT: s_cbranch_scc1
// RELAX-NEXT: s_add_pc_i64
cbranch:
  s_cbranch_scc0 LBB0_0

// RELAX:      [[#%.16x,BR+20]] <end>:
end:
  s_endpgm
