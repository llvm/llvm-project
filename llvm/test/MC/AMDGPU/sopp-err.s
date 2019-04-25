// RUN: not llvm-mc -arch=amdgcn -show-encoding %s 2>&1 | FileCheck --check-prefix=GCN --check-prefix=SICI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti -show-encoding %s 2>&1 | FileCheck --check-prefix=GCN --check-prefix=SICI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=fiji -show-encoding %s 2>&1 | FileCheck --check-prefix=GCN --check-prefix=VI --check-prefix=SICIVI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -show-encoding %s 2>&1 | FileCheck --check-prefix=GCN %s

s_sendmsg sendmsg(11)
// GCN: error: invalid/unsupported code of message

s_sendmsg sendmsg(MSG_INTERRUPTX)
// GCN: error: invalid/unsupported symbolic name of message

s_sendmsg sendmsg(MSG_INTERRUPT, 0)
// GCN: error: failed parsing operand

s_sendmsg sendmsg(MSG_GS)
// GCN: error: failed parsing operand

s_sendmsg sendmsg(MSG_GS, GS_OP_NOP)
// GCN: error: invalid GS_OP: NOP is for GS_DONE only

s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 0, 0)
// GCN: error: failed parsing operand

s_sendmsg sendmsg(MSG_GSX, GS_OP_CUT, 0)
// GCN: error: invalid/unsupported symbolic name of message

s_sendmsg sendmsg(MSG_GS, GS_OP_CUTX, 0)
// GCN: error: invalid symbolic name of GS_OP

s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 4)
// GCN: error: invalid stream id: only 2-bit values are legal

s_sendmsg sendmsg(2)
// GCN: error: failed parsing operand

s_sendmsg sendmsg(2, 0)
// GCN: error: invalid GS_OP: NOP is for GS_DONE only

s_sendmsg sendmsg(2, 3, 0, 0)
// GCN: error: failed parsing operand

s_sendmsg sendmsg(2, 4, 1)
// GCN: error: invalid code of GS_OP: only 2-bit values are legal

s_sendmsg sendmsg(2, 2, 4)
// GCN: error: invalid stream id: only 2-bit values are legal

s_sendmsg sendmsg(2, 2, 0, 0)
// GCN: error: failed parsing operand

s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP, 0)
// GCN: error: failed parsing operand

s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)
// SICIVI: error: invalid/unsupported symbolic name of message

s_sendmsg sendmsg(MSG_GS_ALLOC_REQ, 0)
// SICIVI: error: invalid/unsupported symbolic name of message

s_sendmsg sendmsg(15)
// GCN: error: failed parsing operand

s_sendmsg sendmsg(15, 1, 0)
// GCN: error: failed parsing operand

s_sendmsg sendmsg(15, 0)
// GCN: error: invalid/unsupported code of SYSMSG_OP

s_sendmsg sendmsg(15, 5)
// GCN: error: invalid/unsupported code of SYSMSG_OP

s_sendmsg sendmsg(MSG_SYSMSG)
// GCN: error: failed parsing operand

s_sendmsg sendmsg(MSG_SYSMSG, SYSMSG_OP_ECC_ERR_INTERRUPT, 0)
// GCN: error: failed parsing operand

s_sendmsg sendmsg(MSG_SYSMSG, 0)
// GCN: error: invalid/unsupported code of SYSMSG_OP

s_sendmsg sendmsg(MSG_SYSMSG, 5)
// GCN: error: invalid/unsupported code of SYSMSG_OP

s_waitcnt lgkmcnt(16)
// SICIVI: error: too large value for lgkmcnt

s_waitcnt lgkmcnt(64)
// GCN: error: too large value for lgkmcnt

s_waitcnt expcnt(8)
// GCN: error: too large value for expcnt

s_waitcnt vmcnt(16)
// SICIVI: error: too large value for vmcnt

s_waitcnt vmcnt(64)
// GFX10: error: too large value for vmcnt

s_waitcnt vmcnt(0xFFFFFFFFFFFF0000)
// GCN: error: too large value for vmcnt

s_waitcnt vmcnt(0), expcnt(0), lgkmcnt(0),
// GCN: error: failed parsing operand

s_waitcnt vmcnt(0) & expcnt(0) & lgkmcnt(0)&
// GCN: error: failed parsing operand
