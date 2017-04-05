// RUN: not llvm-mc -arch=amdgcn -show-encoding %s | FileCheck --check-prefix=GCN --check-prefix=SICI %s
// RUN: not llvm-mc -arch=amdgcn -show-encoding %s 2>&1 | FileCheck %s --check-prefix=NOSICI
// RUN: llvm-mc -arch=amdgcn -mcpu=fiji -show-encoding %s | FileCheck --check-prefix=GCN --check-prefix=VI %s

//===----------------------------------------------------------------------===//
// Edge Cases
//===----------------------------------------------------------------------===//

s_nop 0
// GCN: s_nop 0 ; encoding: [0x00,0x00,0x80,0xbf]

s_nop 0xffff
// GCN: s_nop 0xffff ; encoding: [0xff,0xff,0x80,0xbf]

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

s_nop 1
// GCN: s_nop 1 ; encoding: [0x01,0x00,0x80,0xbf]

s_endpgm
// GCN: s_endpgm ; encoding: [0x00,0x00,0x81,0xbf]

s_branch 2
// GCN: s_branch 2 ; encoding: [0x02,0x00,0x82,0xbf]

s_cbranch_scc0 3
// GCN: s_cbranch_scc0 3 ; encoding: [0x03,0x00,0x84,0xbf]

s_cbranch_scc1 4
// GCN: s_cbranch_scc1 4 ; encoding: [0x04,0x00,0x85,0xbf]

s_cbranch_vccz 5
// GCN: s_cbranch_vccz 5 ; encoding: [0x05,0x00,0x86,0xbf]

s_cbranch_vccnz 6
// GCN: s_cbranch_vccnz 6 ; encoding: [0x06,0x00,0x87,0xbf]

s_cbranch_execz 7
// GCN: s_cbranch_execz 7 ; encoding: [0x07,0x00,0x88,0xbf]

s_cbranch_execnz 8
// GCN: s_cbranch_execnz 8 ; encoding: [0x08,0x00,0x89,0xbf]

s_cbranch_cdbgsys 9
// GCN: s_cbranch_cdbgsys 9 ; encoding: [0x09,0x00,0x97,0xbf]

s_cbranch_cdbgsys_and_user 10
// GCN: s_cbranch_cdbgsys_and_user 10 ; encoding: [0x0a,0x00,0x9a,0xbf]

s_cbranch_cdbgsys_or_user 11
// GCN: s_cbranch_cdbgsys_or_user 11 ; encoding: [0x0b,0x00,0x99,0xbf]

s_cbranch_cdbguser 12
// GCN: s_cbranch_cdbguser 12 ; encoding: [0x0c,0x00,0x98,0xbf]

s_barrier
// GCN: s_barrier ; encoding: [0x00,0x00,0x8a,0xbf]

//===----------------------------------------------------------------------===//
// s_waitcnt
//===----------------------------------------------------------------------===//

s_waitcnt 0
// GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]

s_waitcnt vmcnt(0) & expcnt(0) & lgkmcnt(0)
// GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]

s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
// GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]

s_waitcnt vmcnt(0), expcnt(0), lgkmcnt(0)
// GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]

s_waitcnt vmcnt(1)
// GCN: s_waitcnt vmcnt(1) ; encoding: [0x71,0x0f,0x8c,0xbf]

s_waitcnt vmcnt(9)
// GCN: s_waitcnt vmcnt(9) ; encoding: [0x79,0x0f,0x8c,0xbf]

s_waitcnt expcnt(2)
// GCN: s_waitcnt expcnt(2) ; encoding: [0x2f,0x0f,0x8c,0xbf]

s_waitcnt lgkmcnt(3)
// GCN: s_waitcnt lgkmcnt(3) ; encoding: [0x7f,0x03,0x8c,0xbf]

s_waitcnt lgkmcnt(9)
// GCN: s_waitcnt lgkmcnt(9) ; encoding: [0x7f,0x09,0x8c,0xbf]

s_waitcnt vmcnt(0), expcnt(0)
// GCN: s_waitcnt vmcnt(0) expcnt(0) ; encoding: [0x00,0x0f,0x8c,0xbf]


s_sethalt 9
// GCN: s_sethalt 9 ; encoding: [0x09,0x00,0x8d,0xbf]

s_setkill 7
// GCN: s_setkill 7 ; encoding: [0x07,0x00,0x8b,0xbf]

s_sleep 10
// GCN: s_sleep 10 ; encoding: [0x0a,0x00,0x8e,0xbf]

s_setprio 1
// GCN: s_setprio 1 ; encoding: [0x01,0x00,0x8f,0xbf]

s_sendmsg 2
// GCN: s_sendmsg 2 ; encoding: [0x02,0x00,0x90,0xbf]

s_sendmsg 0x1
// GCN: s_sendmsg sendmsg(MSG_INTERRUPT) ; encoding: [0x01,0x00,0x90,0xbf]

s_sendmsg sendmsg(1)
// GCN: s_sendmsg sendmsg(MSG_INTERRUPT) ; encoding: [0x01,0x00,0x90,0xbf]

s_sendmsg sendmsg(MSG_INTERRUPT)
// GCN: s_sendmsg sendmsg(MSG_INTERRUPT) ; encoding: [0x01,0x00,0x90,0xbf]

s_sendmsg 0x12
// GCN: s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 0) ; encoding: [0x12,0x00,0x90,0xbf]

s_sendmsg sendmsg(2, 1)
// GCN: s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 0) ; encoding: [0x12,0x00,0x90,0xbf]

s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 0)
// GCN: s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 0) ; encoding: [0x12,0x00,0x90,0xbf]

s_sendmsg 0x122
// GCN: s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 1) ; encoding: [0x22,0x01,0x90,0xbf]

s_sendmsg sendmsg(2, 2, 1)
// GCN: s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 1) ; encoding: [0x22,0x01,0x90,0xbf]

s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 1)
// GCN: s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 1) ; encoding: [0x22,0x01,0x90,0xbf]

s_sendmsg 0x232
// GCN: s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT_CUT, 2) ; encoding: [0x32,0x02,0x90,0xbf]

s_sendmsg sendmsg(2, 3, 2)
// GCN: s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT_CUT, 2) ; encoding: [0x32,0x02,0x90,0xbf]

s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT_CUT, 2)
// GCN: s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT_CUT, 2) ; encoding: [0x32,0x02,0x90,0xbf]

s_sendmsg 0x3
// GCN: s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP) ; encoding: [0x03,0x00,0x90,0xbf]

s_sendmsg sendmsg(3, 0)
// GCN: s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP) ; encoding: [0x03,0x00,0x90,0xbf]

s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP)
// GCN: s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP) ; encoding: [0x03,0x00,0x90,0xbf]

s_sendmsg 0x4
// GCN: s_sendmsg 4 ; encoding: [0x04,0x00,0x90,0xbf]

s_sendmsg 11
// GCN: s_sendmsg 11 ; encoding: [0x0b,0x00,0x90,0xbf]

s_sendmsg 0x1f
// GCN: s_sendmsg sendmsg(MSG_SYSMSG, SYSMSG_OP_ECC_ERR_INTERRUPT) ; encoding: [0x1f,0x00,0x90,0xbf]

s_sendmsg sendmsg(15, 1)
// GCN: s_sendmsg sendmsg(MSG_SYSMSG, SYSMSG_OP_ECC_ERR_INTERRUPT) ; encoding: [0x1f,0x00,0x90,0xbf]

s_sendmsg sendmsg(MSG_SYSMSG, SYSMSG_OP_ECC_ERR_INTERRUPT)
// GCN: s_sendmsg sendmsg(MSG_SYSMSG, SYSMSG_OP_ECC_ERR_INTERRUPT) ; encoding: [0x1f,0x00,0x90,0xbf]

s_sendmsg 0x6f
// GCN: s_sendmsg 111 ; encoding: [0x6f,0x00,0x90,0xbf]

s_sendmsghalt 3
// GCN: s_sendmsghalt sendmsg(MSG_GS_DONE, GS_OP_NOP) ; encoding: [0x03,0x00,0x91,0xbf]

s_sendmsghalt sendmsg(MSG_GS, GS_OP_EMIT, 1)
// GCN: s_sendmsghalt sendmsg(MSG_GS, GS_OP_EMIT, 1) ; encoding: [0x22,0x01,0x91,0xbf]

s_trap 4
// GCN: s_trap 4 ; encoding: [0x04,0x00,0x92,0xbf]

s_icache_inv
// GCN: s_icache_inv ; encoding: [0x00,0x00,0x93,0xbf]

s_incperflevel 5
// GCN: s_incperflevel 5 ; encoding: [0x05,0x00,0x94,0xbf]

s_decperflevel 6
// GCN: s_decperflevel 6 ; encoding: [0x06,0x00,0x95,0xbf]

s_ttracedata
// GCN: s_ttracedata ; encoding: [0x00,0x00,0x96,0xbf]

s_set_gpr_idx_off
// VI: 	s_set_gpr_idx_off ; encoding: [0x00,0x00,0x9c,0xbf]
// NOSICI: error: instruction not supported on this GPU

s_set_gpr_idx_mode 0
// VI: s_set_gpr_idx_mode 0 ; encoding: [0x00,0x00,0x9d,0xbf]
// NOSICI: error: instruction not supported on this GPU

s_set_gpr_idx_mode 15
// VI: s_set_gpr_idx_mode dst src0 src1 src2 ; encoding: [0x0f,0x00,0x9d,0xbf]
// NOSICI: error: instruction not supported on this GPU
