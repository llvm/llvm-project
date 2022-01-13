// RUN: not llvm-mc -arch=amdgcn %s 2>&1 | FileCheck --check-prefixes=GCN,PREGFX11,SICI --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti %s 2>&1 | FileCheck --check-prefixes=GCN,PREGFX11,SICI --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=fiji %s 2>&1 | FileCheck --check-prefixes=GCN,PREGFX11,VI --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 %s 2>&1 | FileCheck --check-prefixes=GCN,PREGFX11,GFX10 --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 %s 2>&1 | FileCheck --check-prefixes=GCN,GFX11 --implicit-check-not=error: %s

//===----------------------------------------------------------------------===//
// sendmsg
//===----------------------------------------------------------------------===//

s_sendmsg sendmsg(MSG_INTERRUPTX)
// GCN: error: expected a message name or an absolute expression

s_sendmsg sendmsg(1 -)
// GCN: error: unknown token in expression

s_sendmsg sendmsg(MSG_INTERRUPT, 0)
// GCN: error: message does not support operations

s_sendmsg sendmsg(MSG_INTERRUPT, 0, 0)
// GCN: error: message does not support operations

s_sendmsg sendmsg(MSG_GS)
// PREGFX11: error: missing message operation
// GFX11: error: expected a message name or an absolute expression

s_sendmsg sendmsg(MSG_GS, GS_OP_NOP)
// PREGFX11: error: invalid operation id
// GFX11: error: expected a message name or an absolute expression

s_sendmsg sendmsg(MSG_GS, SYSMSG_OP_ECC_ERR_INTERRUPT)
// PREGFX11: error: expected an operation name or an absolute expression
// GFX11: error: expected a message name or an absolute expression

s_sendmsg sendmsg(MSG_GS, 0)
// PREGFX11: error: invalid operation id
// GFX11: error: expected a message name or an absolute expression

s_sendmsg sendmsg(MSG_GS, -1)
// PREGFX11: error: invalid operation id
// GFX11: error: expected a message name or an absolute expression

s_sendmsg sendmsg(MSG_GS, 4)
// PREGFX11: error: invalid operation id
// GFX11: error: expected a message name or an absolute expression

s_sendmsg sendmsg(MSG_GS, 8)
// PREGFX11: error: invalid operation id
// GFX11: error: expected a message name or an absolute expression

s_sendmsg sendmsg(15, -1)
// GCN: error: invalid operation id

s_sendmsg sendmsg(15, 8)
// GCN: error: invalid operation id

s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 0, 0)
// PREGFX11: error: expected a closing parenthesis
// GFX11: error: expected a message name or an absolute expression

s_sendmsg sendmsg(MSG_GSX, GS_OP_CUT, 0)
// GCN: error: expected a message name or an absolute expression

s_sendmsg sendmsg(MSG_GS, GS_OP_CUTX, 0)
// PREGFX11: error: expected an operation name or an absolute expression
// GFX11: error: expected a message name or an absolute expression

s_sendmsg sendmsg(MSG_GS, 1 -)
// PREGFX11: error: unknown token in expression
// GFX11: error: expected a message name or an absolute expression

s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 4)
// PREGFX11: error: invalid message stream id
// GFX11: error: expected a message name or an absolute expression

s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 1 -)
// PREGFX11: error: unknown token in expression
// GFX11: error: expected a message name or an absolute expression

s_sendmsg sendmsg(2, 3, 0, 0)
// GCN: error: expected a closing parenthesis

s_sendmsg sendmsg(2, 2, -1)
// GCN: error: invalid message stream id

s_sendmsg sendmsg(2, 2, 4)
// GCN: error: invalid message stream id

s_sendmsg sendmsg(2, 2, 0, 0)
// GCN: error: expected a closing parenthesis

s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP, 0)
// PREGFX11: error: message operation does not support streams
// GFX11: error: expected a message name or an absolute expression

s_sendmsg sendmsg(MSG_GS_DONE, 0, 0)
// PREGFX11: error: message operation does not support streams
// GFX11: error: expected a message name or an absolute expression

s_sendmsg sendmsg(MSG_HS_TESSFACTOR)
// SICI: error: expected a message name or an absolute expression
// VI: error: expected a message name or an absolute expression
// GFX10: error: expected a message name or an absolute expression

s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
// SICI: error: expected a message name or an absolute expression
// VI: error: expected a message name or an absolute expression
// GFX10: error: expected a message name or an absolute expression

s_sendmsg sendmsg(MSG_SAVEWAVE)
// SICI: error: invalid message id

s_sendmsg sendmsg(MSG_STALL_WAVE_GEN)
// SICI: error: invalid message id
// VI: error: invalid message id

s_sendmsg sendmsg(MSG_HALT_WAVES)
// SICI: error: invalid message id
// VI: error: invalid message id

s_sendmsg sendmsg(MSG_ORDERED_PS_DONE)
// SICI: error: invalid message id
// VI: error: invalid message id

s_sendmsg sendmsg(MSG_EARLY_PRIM_DEALLOC)
// SICI: error: invalid message id
// VI: error: invalid message id
// GFX10: error: invalid message id
// GFX11: error: expected a message name or an absolute expression

s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)
// VI: error: invalid message id
// SICI: error: invalid message id

s_sendmsg sendmsg(MSG_GS_ALLOC_REQ, 0)
// VI: error: invalid message id
// SICI: error: invalid message id
// GFX10: error: message does not support operations
// GFX11: error: message does not support operations

s_sendmsg sendmsg(MSG_GET_DOORBELL)
// SICI: error: invalid message id
// VI: error: invalid message id

s_sendmsg sendmsg(MSG_GET_DDID)
// SICI: error: invalid message id
// VI: error: invalid message id

s_sendmsg sendmsg(MSG_GET_TMA)
// SICI: error: invalid message id
// VI: error: invalid message id
// GFX10: error: invalid message id

s_sendmsg sendmsg(MSG_GET_REALTIME)
// SICI: error: invalid message id
// VI: error: invalid message id
// GFX10: error: invalid message id

s_sendmsg sendmsg(MSG_GET_TBA)
// SICI: error: invalid message id
// VI: error: invalid message id
// GFX10: error: invalid message id

s_sendmsg sendmsg(-1)
// GCN: error: invalid message id

s_sendmsg sendmsg(16)
// GCN: error: invalid message id

s_sendmsg sendmsg(MSG_SYSMSG)
// GCN: error: missing message operation

s_sendmsg sendmsg(MSG_SYSMSG, SYSMSG_OP_ECC_ERR_INTERRUPT, 0)
// GCN: error: message operation does not support streams

s_sendmsg sendmsg(MSG_SYSMSG, 0)
// GCN: error: invalid operation id

s_sendmsg sendmsg(MSG_SYSMSG, 5)
// GCN: error: invalid operation id

//===----------------------------------------------------------------------===//
// waitcnt
//===----------------------------------------------------------------------===//

s_waitcnt lgkmcnt(16)
// VI: error: too large value for lgkmcnt
// SICI: error: too large value for lgkmcnt

s_waitcnt lgkmcnt(64)
// GCN: error: too large value for lgkmcnt

s_waitcnt expcnt(8)
// PREGFX11: error: too large value for expcnt

s_waitcnt expcnt(16)
// GCN: error: too large value for expcnt

s_waitcnt vmcnt(16)
// VI: error: too large value for vmcnt
// SICI: error: too large value for vmcnt

s_waitcnt vmcnt(64)
// GCN: error: too large value for vmcnt

s_waitcnt vmcnt(0xFFFFFFFFFFFF0000)
// GCN: error: too large value for vmcnt

s_waitcnt vmcnt(0), expcnt(0), lgkmcnt(0),
// GCN: error: expected a counter name

s_waitcnt vmcnt(0) & expcnt(0) & lgkmcnt(0)&
// GCN: error: expected a counter name

s_waitcnt vmcnt(0) & expcnt(0) & x
// GCN: error: expected a left parenthesis

s_waitcnt vmcnt(0) & expcnt(0) x
// GCN: error: expected a left parenthesis

s_waitcnt vmcnt(0) & expcnt(0) & 1
// GCN: error: expected a counter name

s_waitcnt vmcnt(0) & expcnt(0) 1
// GCN: error: expected a counter name

s_waitcnt vmcnt(0) & expcnt(0) x(0)
// GCN: error: invalid counter name x

s_waitcnt vmcnt(x)
// GCN: error: expected absolute expression

s_waitcnt x
// GCN: error: expected absolute expression

s_waitcnt vmcnt(0
// GCN: error: expected a closing parenthesis

s_branch 0x80000000ffff
// GCN: error: expected a 16-bit signed jump offset

s_branch 0x10000
// GCN: error: expected a 16-bit signed jump offset

s_branch -32769
// GCN: error: expected a 16-bit signed jump offset

s_branch 1.0
// GCN: error: expected a 16-bit signed jump offset

s_branch s0
// GCN: error: invalid operand for instruction

s_branch offset:1
// GCN: error: not a valid operand
