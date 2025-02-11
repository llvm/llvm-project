// RUN: not llvm-mc -triple=amdgcn -show-encoding < %s | FileCheck %s
// RUN: not llvm-mc -triple=amdgcn -show-encoding -filetype=null %s 2>&1 | FileCheck -check-prefix=ERR %s
// RUN: llvm-mc -triple=amdgcn -mcpu=tahiti -show-encoding < %s | FileCheck %s

// CHECK: v_cmp_lt_f32_e32 vcc, s2, v4            ; encoding: [0x02,0x08,0x02,0x7c]
v_cmp_lt_f32 vcc, s2, v4

// CHECK: v_cndmask_b32_e32 v1, v2, v3, vcc       ; encoding: [0x02,0x07,0x02,0x00]
v_cndmask_b32 v1, v2, v3, vcc

// ERR: [[@LINE+1]]:1: error: instruction not supported on this GPU
v_mac_legacy_f32 v1, v3, s5

// CHECK: v_lshr_b32_e32 v0, v1, v2               ; encoding: [0x01,0x05,0x00,0x2a]
v_lshr_b32 v0, v1, v2
