// RUN: llvm-mc -triple=amdgcn -mcpu=tahiti -show-encoding %s | FileCheck %s --check-prefix=SICI
// RUN: llvm-mc -triple=amdgcn -mcpu=tonga -show-encoding %s | FileCheck %s --check-prefix=VI
// XFAIL: *

// FIXME: We should be printing _e64 suffixes for these. 
// FIXME: When this is fixed delete this file and fix test case in vop3.s

v_nop_e64
// SICI: v_nop_e64 ; encoding: [0x00,0x00,0x00,0xd3,0x00,0x00,0x00,0x00]
// VI:   v_nop_e64 ; encoding: [0x00,0x00,0x40,0xd1,0x00,0x00,0x00,0x00]

v_clrexcp_e64
// SICI: v_clrexcp_e64 ; encoding: [0x00,0x00,0x82,0xd3,0x00,0x00,0x00,0x00]
// VI:   v_clrexcp_e64 ; encoding: [0x00,0x00,0x75,0xd1,0x00,0x00,0x00,0x00]
