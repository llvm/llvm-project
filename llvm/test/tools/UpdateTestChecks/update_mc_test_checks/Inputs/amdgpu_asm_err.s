// RUN: not llvm-mc -triple=amdgcn -show-encoding %s 2>&1 | FileCheck --check-prefixes=CHECKA %s
// RUN: not llvm-mc -triple=amdgcn %s 2>&1 | FileCheck --check-prefixes=CHECKB %s

v_bfrev_b32 v5, v299

v_bfrev_b32 v5, v1
