# RUN: not llvm-mc -triple riscv32 < %s 2>&1 | FileCheck %s

# Too many operands
.insn i  0x13,  0,  a0, a1, 13, 14 # CHECK: :[[@LINE]]:33: error: invalid operand for instruction
.insn r  0x43,  0,  0, fa0, fa1, fa2, fa3, fa4 # CHECK: :[[@LINE]]:44: error: invalid operand for instruction

# Too few operands
.insn r  0x33,  0,  0, a0, a1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
.insn i  0x13,  0,  a0, a1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction

.insn r  0x33,  0,  0, a0, 13 # CHECK: :[[@LINE]]:28: error: invalid operand for instruction
.insn i  0x13,  0, a0, a1, a2 # CHECK: :[[@LINE]]:28: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an integer in the range [-2048, 2047]

.insn q  0x13,  0,  a0, a1, 13, 14 # CHECK: :[[@LINE]]:7: error: invalid instruction format

# Invalid immediate
.insn i  0x99,  0, a0, 4(a1) # CHECK: :[[@LINE]]:10: error: opcode must be a valid opcode name or an immediate in the range [0, 127]
.insn r  0x33,  8,  0, a0, a1, a2 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 7]
.insn r4 0x43,  0,  4, fa0, fa1, fa2, fa3 # CHECK: :[[@LINE]]:21: error: immediate must be an integer in the range [0, 3]

# Unrecognized opcode name
.insn r UNKNOWN, 0, a1, a2, a3 # CHECK: :[[@LINE]]:9: error: opcode must be a valid opcode name or an immediate in the range [0, 127]

# Make fake mnemonics we use to match these in the tablegened asm match table isn't exposed.
.insn_i  0x13,  0,  a0, a1, 13, 14 # CHECK: :[[@LINE]]:1: error: unknown directive
