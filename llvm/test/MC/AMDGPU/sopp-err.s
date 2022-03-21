// RUN: not llvm-mc -arch=amdgcn %s 2>&1 | FileCheck --check-prefixes=GCN,SICI --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti %s 2>&1 | FileCheck --check-prefixes=GCN,SICI --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=fiji %s 2>&1 | FileCheck --check-prefixes=GCN,VI --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 %s 2>&1 | FileCheck --check-prefixes=GCN,GFX10 --implicit-check-not=error: %s

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
// GCN: error: missing message operation

s_sendmsg sendmsg(MSG_GS, GS_OP_NOP)
// GCN: error: invalid operation id

s_sendmsg sendmsg(MSG_GS, SYSMSG_OP_ECC_ERR_INTERRUPT)
// GCN: error: expected an operation name or an absolute expression

s_sendmsg sendmsg(MSG_GS, 0)
// GCN: error: invalid operation id

s_sendmsg sendmsg(MSG_GS, -1)
// GCN: error: invalid operation id

s_sendmsg sendmsg(MSG_GS, 4)
// GCN: error: invalid operation id

s_sendmsg sendmsg(MSG_GS, 8)
// GCN: error: invalid operation id

s_sendmsg sendmsg(15, -1)
// GCN: error: invalid operation id

s_sendmsg sendmsg(15, 8)
// GCN: error: invalid operation id

s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 0, 0)
// GCN: error: expected a closing parenthesis

s_sendmsg sendmsg(MSG_GSX, GS_OP_CUT, 0)
// GCN: error: expected a message name or an absolute expression

s_sendmsg sendmsg(MSG_GS, GS_OP_CUTX, 0)
// GCN: error: expected an operation name or an absolute expression

s_sendmsg sendmsg(MSG_GS, 1 -)
// GCN: error: unknown token in expression

s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 4)
// GCN: error: invalid message stream id

s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 1 -)
// GCN: error: unknown token in expression

s_sendmsg sendmsg(2, 3, 0, 0)
// GCN: error: expected a closing parenthesis

s_sendmsg sendmsg(2, 2, -1)
// GCN: error: invalid message stream id

s_sendmsg sendmsg(2, 2, 4)
// GCN: error: invalid message stream id

s_sendmsg sendmsg(2, 2, 0, 0)
// GCN: error: expected a closing parenthesis

s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP, 0)
// GCN: error: message operation does not support streams

s_sendmsg sendmsg(MSG_GS_DONE, 0, 0)
// GCN: error: message operation does not support streams

s_sendmsg sendmsg(MSG_SAVEWAVE)
// SICI: error: specified message id is not supported on this GPU

s_sendmsg sendmsg(MSG_STALL_WAVE_GEN)
// SICI: error: specified message id is not supported on this GPU
// VI: error: specified message id is not supported on this GPU

s_sendmsg sendmsg(MSG_HALT_WAVES)
// SICI: error: specified message id is not supported on this GPU
// VI: error: specified message id is not supported on this GPU

s_sendmsg sendmsg(MSG_ORDERED_PS_DONE)
// SICI: error: specified message id is not supported on this GPU
// VI: error: specified message id is not supported on this GPU

s_sendmsg sendmsg(MSG_EARLY_PRIM_DEALLOC)
// SICI: error: specified message id is not supported on this GPU
// VI: error: specified message id is not supported on this GPU
// GFX10: error: specified message id is not supported on this GPU

s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)
// VI: error: specified message id is not supported on this GPU
// SICI: error: specified message id is not supported on this GPU

s_sendmsg sendmsg(MSG_GS_ALLOC_REQ, 0)
// VI: error: specified message id is not supported on this GPU
// SICI: error: specified message id is not supported on this GPU
// GFX10: error: message does not support operations

s_sendmsg sendmsg(MSG_GET_DOORBELL)
// SICI: error: specified message id is not supported on this GPU
// VI: error: specified message id is not supported on this GPU

s_sendmsg sendmsg(MSG_GET_DDID)
// SICI: error: specified message id is not supported on this GPU
// VI: error: specified message id is not supported on this GPU

s_sendmsg sendmsg(-1)
// VI: error: invalid message id
// SICI: error: invalid message id
// GFX10: error: invalid message id

s_sendmsg sendmsg(16)
// VI: error: invalid message id
// SICI: error: invalid message id
// GFX10: error: invalid message id

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
// GCN: error: too large value for expcnt

s_waitcnt vmcnt(16)
// VI: error: too large value for vmcnt
// SICI: error: too large value for vmcnt

s_waitcnt vmcnt(64)
// GFX10: error: too large value for vmcnt
// SICI: error: too large value for vmcnt
// VI: error: too large value for vmcnt

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
