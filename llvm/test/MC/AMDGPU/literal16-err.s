// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga %s 2>&1 | FileCheck -check-prefix=NOVI --implicit-check-not=error: %s

v_add_f16 v1, 0xfffff, v2
// NOVI: error: invalid operand for instruction

v_add_f16 v1, 0x10000, v2
// NOVI: error: invalid operand for instruction

v_add_f16 v1, 0xffffffffffff000f, v2
// NOVI: error: invalid operand for instruction

v_add_f16 v1, 0x1000ffff, v2
// NOVI: error: invalid operand for instruction

v_add_f16 v1, -32769, v2
// NOVI: error: invalid operand for instruction

v_add_f16 v1, 65536, v2
// NOVI: error: invalid operand for instruction

v_add_f32 v1, 4294967296, v2
// NOVI: error: invalid operand for instruction

v_add_f32 v1, 0x0000000100000000, v2
// NOVI: error: invalid operand for instruction

v_and_b32 v1, 0x0000000100000000, v2
// NOVI: error: invalid operand for instruction
