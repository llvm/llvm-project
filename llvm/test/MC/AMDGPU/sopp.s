// RUN: not llvm-mc -arch=amdgcn -show-encoding %s | FileCheck --check-prefixes=GCN,SI %s
// RUN: not llvm-mc -arch=amdgcn %s 2>&1 | FileCheck %s --check-prefix=NOSICI --implicit-check-not=error:
// RUN: llvm-mc -arch=amdgcn -mcpu=fiji -show-encoding %s | FileCheck --check-prefixes=GCN,VI %s

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

s_waitcnt vmcnt(15)
// GCN: s_waitcnt vmcnt(15) expcnt(7) lgkmcnt(15) ; encoding: [0x7f,0x0f,0x8c,0xbf]

s_waitcnt vmcnt_sat(9)
// GCN: s_waitcnt vmcnt(9) ; encoding: [0x79,0x0f,0x8c,0xbf]

s_waitcnt vmcnt_sat(15)
// GCN: s_waitcnt vmcnt(15) expcnt(7) lgkmcnt(15) ; encoding: [0x7f,0x0f,0x8c,0xbf]

s_waitcnt vmcnt_sat(16)
// GCN: s_waitcnt vmcnt(15) expcnt(7) lgkmcnt(15) ; encoding: [0x7f,0x0f,0x8c,0xbf]

s_waitcnt expcnt(2)
// GCN: s_waitcnt expcnt(2) ; encoding: [0x2f,0x0f,0x8c,0xbf]

s_waitcnt expcnt(7)
// GCN: s_waitcnt vmcnt(15) expcnt(7) lgkmcnt(15) ; encoding: [0x7f,0x0f,0x8c,0xbf]

s_waitcnt expcnt_sat(2)
// GCN: s_waitcnt expcnt(2) ; encoding: [0x2f,0x0f,0x8c,0xbf]

s_waitcnt expcnt_sat(7)
// GCN: s_waitcnt vmcnt(15) expcnt(7) lgkmcnt(15) ; encoding: [0x7f,0x0f,0x8c,0xbf]

s_waitcnt expcnt_sat(0xFFFF0000)
// GCN: s_waitcnt vmcnt(15) expcnt(7) lgkmcnt(15) ; encoding: [0x7f,0x0f,0x8c,0xbf]

s_waitcnt lgkmcnt(3)
// GCN: s_waitcnt lgkmcnt(3) ; encoding: [0x7f,0x03,0x8c,0xbf]

s_waitcnt lgkmcnt(9)
// GCN: s_waitcnt lgkmcnt(9) ; encoding: [0x7f,0x09,0x8c,0xbf]

s_waitcnt lgkmcnt(15)
// GCN: s_waitcnt vmcnt(15) expcnt(7) lgkmcnt(15) ; encoding: [0x7f,0x0f,0x8c,0xbf]

s_waitcnt vmcnt(0), expcnt(0)
// GCN: s_waitcnt vmcnt(0) expcnt(0) ; encoding: [0x00,0x0f,0x8c,0xbf]

s_waitcnt lgkmcnt_sat(3)
// GCN: s_waitcnt lgkmcnt(3) ; encoding: [0x7f,0x03,0x8c,0xbf]

s_waitcnt lgkmcnt_sat(9)
// GCN: s_waitcnt lgkmcnt(9) ; encoding: [0x7f,0x09,0x8c,0xbf]

s_waitcnt lgkmcnt_sat(15)
// GCN: s_waitcnt vmcnt(15) expcnt(7) lgkmcnt(15) ; encoding: [0x7f,0x0f,0x8c,0xbf]

s_waitcnt lgkmcnt_sat(16)
// GCN: s_waitcnt vmcnt(15) expcnt(7) lgkmcnt(15) ; encoding: [0x7f,0x0f,0x8c,0xbf]

x=1
s_waitcnt lgkmcnt_sat(x+1)
// GCN: s_waitcnt lgkmcnt(2)            ; encoding: [0x7f,0x02,0x8c,0xbf]

s_waitcnt lgkmcnt_sat(1+x)
// GCN: s_waitcnt lgkmcnt(2)            ; encoding: [0x7f,0x02,0x8c,0xbf]

s_waitcnt x+1
// GCN: s_waitcnt vmcnt(2) expcnt(0) lgkmcnt(0) ; encoding: [0x02,0x00,0x8c,0xbf]

s_waitcnt 1+x
// GCN: s_waitcnt vmcnt(2) expcnt(0) lgkmcnt(0) ; encoding: [0x02,0x00,0x8c,0xbf]

lgkmcnt_sat=1
s_waitcnt lgkmcnt_sat
// GCN: s_waitcnt vmcnt(1) expcnt(0) lgkmcnt(0) ; encoding: [0x01,0x00,0x8c,0xbf]

s_waitcnt lgkmcnt_sat+1
// GCN: s_waitcnt vmcnt(2) expcnt(0) lgkmcnt(0) ; encoding: [0x02,0x00,0x8c,0xbf]

//===----------------------------------------------------------------------===//
// misc sopp instructions
//===----------------------------------------------------------------------===//

s_sethalt 9
// GCN: s_sethalt 9 ; encoding: [0x09,0x00,0x8d,0xbf]

s_setkill 7
// GCN: s_setkill 7 ; encoding: [0x07,0x00,0x8b,0xbf]

s_sleep 10
// GCN: s_sleep 10 ; encoding: [0x0a,0x00,0x8e,0xbf]

s_setprio 1
// GCN: s_setprio 1 ; encoding: [0x01,0x00,0x8f,0xbf]

//===----------------------------------------------------------------------===//
// s_sendmsg
//===----------------------------------------------------------------------===//

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

s_sendmsg sendmsg(2, GS_OP_CUT)
// GCN: s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 0) ; encoding: [0x12,0x00,0x90,0xbf]

s_sendmsg sendmsg(MSG_GS, GS_OP_CUT)
// GCN: s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 0) ; encoding: [0x12,0x00,0x90,0xbf]

s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 0)
// GCN: s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 0) ; encoding: [0x12,0x00,0x90,0xbf]

s_sendmsg sendmsg(MSG_GS, 1)
// GCN: s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 0) ; encoding: [0x12,0x00,0x90,0xbf]

s_sendmsg sendmsg(MSG_GS, 1, 1)
// GCN: s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 1) ; encoding: [0x12,0x01,0x90,0xbf]

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
// SI: s_sendmsg sendmsg(4, 0, 0) ; encoding: [0x04,0x00,0x90,0xbf]
// VI: s_sendmsg sendmsg(MSG_SAVEWAVE) ; encoding: [0x04,0x00,0x90,0xbf]

s_sendmsg sendmsg(4)
// SI: s_sendmsg sendmsg(4, 0, 0) ; encoding: [0x04,0x00,0x90,0xbf]
// VI: s_sendmsg sendmsg(MSG_SAVEWAVE) ; encoding: [0x04,0x00,0x90,0xbf]

s_sendmsg sendmsg(MSG_SAVEWAVE)
// NOSICI: error: specified message id is not supported on this GPU
// VI: s_sendmsg sendmsg(MSG_SAVEWAVE) ; encoding: [0x04,0x00,0x90,0xbf]

s_sendmsg 0x1f
// GCN: s_sendmsg sendmsg(MSG_SYSMSG, SYSMSG_OP_ECC_ERR_INTERRUPT) ; encoding: [0x1f,0x00,0x90,0xbf]

s_sendmsg sendmsg(15, 1)
// GCN: s_sendmsg sendmsg(MSG_SYSMSG, SYSMSG_OP_ECC_ERR_INTERRUPT) ; encoding: [0x1f,0x00,0x90,0xbf]

s_sendmsg sendmsg(MSG_SYSMSG, SYSMSG_OP_ECC_ERR_INTERRUPT)
// GCN: s_sendmsg sendmsg(MSG_SYSMSG, SYSMSG_OP_ECC_ERR_INTERRUPT) ; encoding: [0x1f,0x00,0x90,0xbf]

s_sendmsghalt 3
// GCN: s_sendmsghalt sendmsg(MSG_GS_DONE, GS_OP_NOP) ; encoding: [0x03,0x00,0x91,0xbf]

s_sendmsghalt sendmsg(MSG_GS, GS_OP_EMIT, 1)
// GCN: s_sendmsghalt sendmsg(MSG_GS, GS_OP_EMIT, 1) ; encoding: [0x22,0x01,0x91,0xbf]

//===----------------------------------------------------------------------===//
// s_sendmsg with a numeric message id (no validation)
//===----------------------------------------------------------------------===//

s_sendmsg 2
// GCN: s_sendmsg sendmsg(2, 0, 0) ; encoding: [0x02,0x00,0x90,0xbf]

s_sendmsg 9
// GCN: s_sendmsg sendmsg(9, 0, 0) ; encoding: [0x09,0x00,0x90,0xbf]

s_sendmsg 11
// GCN: s_sendmsg sendmsg(11, 0, 0) ; encoding: [0x0b,0x00,0x90,0xbf]

s_sendmsg 0x6f
// GCN: s_sendmsg sendmsg(15, 6, 0) ; encoding: [0x6f,0x00,0x90,0xbf]

s_sendmsg sendmsg(1, 3)
// GCN: s_sendmsg sendmsg(1, 3, 0)      ; encoding: [0x31,0x00,0x90,0xbf]

s_sendmsg sendmsg(1, 3, 2)
// GCN: s_sendmsg sendmsg(1, 3, 2)      ; encoding: [0x31,0x02,0x90,0xbf]

s_sendmsg sendmsg(2, 0, 1)
// GCN: s_sendmsg sendmsg(2, 0, 1)      ; encoding: [0x02,0x01,0x90,0xbf]

s_sendmsg sendmsg(15, 7, 3)
// GCN: s_sendmsg sendmsg(15, 7, 3)     ; encoding: [0x7f,0x03,0x90,0xbf]

s_sendmsg 4567
// GCN: s_sendmsg 4567                  ; encoding: [0xd7,0x11,0x90,0xbf]

//===----------------------------------------------------------------------===//
// s_sendmsg with expressions
//===----------------------------------------------------------------------===//

sendmsg=2
s_sendmsg sendmsg
// GCN: s_sendmsg sendmsg(2, 0, 0) ; encoding: [0x02,0x00,0x90,0xbf]

sendmsg=1
s_sendmsg sendmsg+1
// GCN: s_sendmsg sendmsg(2, 0, 0) ; encoding: [0x02,0x00,0x90,0xbf]

s_sendmsg 1+sendmsg
// GCN: s_sendmsg sendmsg(2, 0, 0) ; encoding: [0x02,0x00,0x90,0xbf]

msg=1
s_sendmsg sendmsg(msg)
// GCN: s_sendmsg sendmsg(MSG_INTERRUPT) ; encoding: [0x01,0x00,0x90,0xbf]

msg=0
s_sendmsg sendmsg(msg+1)
// GCN: s_sendmsg sendmsg(MSG_INTERRUPT) ; encoding: [0x01,0x00,0x90,0xbf]

msg=0
s_sendmsg sendmsg(1+msg)
// GCN: s_sendmsg sendmsg(MSG_INTERRUPT) ; encoding: [0x01,0x00,0x90,0xbf]

msg=2
op=1
s_sendmsg sendmsg(msg, op)
// GCN: s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 0) ; encoding: [0x12,0x00,0x90,0xbf]

msg=1
op=0
s_sendmsg sendmsg(msg+1, op+1)
// GCN: s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 0) ; encoding: [0x12,0x00,0x90,0xbf]

msg=1
op=0
s_sendmsg sendmsg(1+msg, 1+op)
// GCN: s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 0) ; encoding: [0x12,0x00,0x90,0xbf]

msg=1
op=2
stream=1
s_sendmsg sendmsg(msg+1, op+1, stream+1)
// GCN: s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT_CUT, 2) ; encoding: [0x32,0x02,0x90,0xbf]

msg=1
op=2
stream=1
s_sendmsg sendmsg(1+msg, 1+op, 1+stream)
// GCN: s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT_CUT, 2) ; encoding: [0x32,0x02,0x90,0xbf]

MSG_GS=-1
GS_OP_EMIT=-1
s_sendmsghalt sendmsg(MSG_GS, GS_OP_EMIT, 1)
// GCN: s_sendmsghalt sendmsg(MSG_GS, GS_OP_EMIT, 1) ; encoding: [0x22,0x01,0x91,0xbf]

//===----------------------------------------------------------------------===//
// misc sopp instructions
//===----------------------------------------------------------------------===//

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
// VI: s_set_gpr_idx_mode gpr_idx() ; encoding: [0x00,0x00,0x9d,0xbf]
// NOSICI: error: instruction not supported on this GPU

s_set_gpr_idx_mode gpr_idx()
// VI: s_set_gpr_idx_mode gpr_idx() ; encoding: [0x00,0x00,0x9d,0xbf]
// NOSICI: error: instruction not supported on this GPU

s_set_gpr_idx_mode 15
// VI: s_set_gpr_idx_mode gpr_idx(SRC0,SRC1,SRC2,DST) ; encoding: [0x0f,0x00,0x9d,0xbf]
// NOSICI: error: instruction not supported on this GPU

s_set_gpr_idx_mode gpr_idx(SRC2,SRC1,SRC0,DST)
// VI: s_set_gpr_idx_mode gpr_idx(SRC0,SRC1,SRC2,DST) ; encoding: [0x0f,0x00,0x9d,0xbf]
// NOSICI: error: instruction not supported on this GPU

s_endpgm_saved
// VI: s_endpgm_saved ; encoding: [0x00,0x00,0x9b,0xbf]
// NOSICI: error: instruction not supported on this GPU

s_wakeup
// VI: s_wakeup ; encoding: [0x00,0x00,0x83,0xbf]
// NOSICI: error: instruction not supported on this GPU

//===----------------------------------------------------------------------===//
// absolute expressions as branch offsets
//===----------------------------------------------------------------------===//

offset = 3
s_branch 1+offset
// GCN: s_branch 4 ; encoding: [0x04,0x00,0x82,0xbf]

offset = 3
s_branch offset+1
// GCN: s_branch 4 ; encoding: [0x04,0x00,0x82,0xbf]
