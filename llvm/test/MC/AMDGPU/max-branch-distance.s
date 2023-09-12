//  RUN: not llvm-mc -triple=amdgcn -filetype=obj -o /dev/null %s 2>&1 | FileCheck -check-prefix=ERROR %s
//  ERROR: max-branch-distance.s:7:3: error: branch size exceeds simm16

// fill v_nop
LBB0_0:
    .fill 32768, 4, 0x0000007e
  s_branch LBB0_0
