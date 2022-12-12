// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga %s 2>&1 | FileCheck -check-prefix=NOVI --implicit-check-not=error: %s

s_memtime exec
// NOVI: :[[@LINE-1]]:11: error: invalid operand for instruction

s_memrealtime exec
// NOVI: :[[@LINE-1]]:15: error: invalid operand for instruction

s_store_dword m0, s[2:3], 0x0
// NOVI: :[[@LINE-1]]:15: error: invalid operand for instruction

s_store_dword exec_lo, s[2:3], 0x0
// NOVI: :[[@LINE-1]]:15: error: invalid operand for instruction

s_store_dword exec_hi, s[2:3], 0x0
// NOVI: :[[@LINE-1]]:15: error: invalid operand for instruction

s_store_dwordx2 exec, s[2:3], 0x0
// NOVI: :[[@LINE-1]]:17: error: invalid operand for instruction

s_buffer_store_dword m0, s[0:3], 0x0
// NOVI: :[[@LINE-1]]:22: error: invalid operand for instruction

s_buffer_store_dword exec_lo, s[0:3], 0x0
// NOVI: :[[@LINE-1]]:22: error: invalid operand for instruction

s_buffer_store_dword exec_hi, s[0:3], 0x0
// NOVI: :[[@LINE-1]]:22: error: invalid operand for instruction

s_buffer_store_dwordx2 exec, s[0:3], 0x0
// NOVI: :[[@LINE-1]]:24: error: invalid operand for instruction

s_load_dword m0, s[0:1], s4
// NOVI: :[[@LINE-1]]:14: error: invalid operand for instruction

s_load_dword exec_lo, s[0:1], s4
// NOVI: :[[@LINE-1]]:14: error: invalid operand for instruction

s_load_dword exec_hi, s[0:1], s4
// NOVI: :[[@LINE-1]]:14: error: invalid operand for instruction

s_load_dwordx2 exec, s[0:1], s4
// NOVI: :[[@LINE-1]]:16: error: invalid operand for instruction

s_buffer_load_dword m0, s[0:3], s4
// NOVI: :[[@LINE-1]]:21: error: invalid operand for instruction

s_buffer_load_dword exec_lo, s[0:3], s4
// NOVI: :[[@LINE-1]]:21: error: invalid operand for instruction

s_buffer_load_dword exec_hi, s[0:3], s4
// NOVI: :[[@LINE-1]]:21: error: invalid operand for instruction

s_buffer_load_dwordx2 exec, s[0:3], s4
// NOVI: :[[@LINE-1]]:23: error: invalid operand for instruction
