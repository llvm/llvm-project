// RUN: llvm-mc -triple=amdgcn -show-encoding %s 2>&1 | FileCheck --check-prefixes=CHECK %s

v_bfrev_b32 v5, v1 //This is comment A

v_bfrev_b32 v1, v1
// This is comment B

// This is comment C
v_bfrev_b32 v2, v1
