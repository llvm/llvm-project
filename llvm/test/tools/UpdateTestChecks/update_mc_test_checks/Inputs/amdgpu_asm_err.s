// RUN: not llvm-mc -triple=amdgcn -show-encoding %s 2>&1 | FileCheck --check-prefixes=CHECK %s

v_bfrev_b32 v5, v299
