# RUN: not llvm-mc -triple riscv32 -mattr=+c < %s 2>&1 | FileCheck --match-full-lines %s

# Too many operands
.insn ci  1, 0, a0, 13, 14 # CHECK: <stdin>:[[#@LINE]]:25: error: unexpected extra operand for instruction
.insn cr  2, 9, a0, a1, a2 # CHECK: <stdin>:[[#@LINE]]:25: error: unexpected extra operand for instruction

## Too few operands
.insn ci  1, 0, a0 # CHECK: <stdin>:[[#@LINE]]:19: error: too few operands for instruction
.insn cr  2, 9, a0 # CHECK: <stdin>:[[#@LINE]]:19: error: too few operands for instruction

.insn cr  2, 9, a0, 13 # CHECK: <stdin>:[[#@LINE]]:21: error: invalid operand for instruction
.insn ci  1, 0, a0, a1 # CHECK: <stdin>:[[#@LINE]]:21: error: immediate must be an integer in the range [-32, 31]

.insn cq  0x13,  0,  a0, a1, 13, 14 # CHECK: <stdin>:[[#@LINE]]:7: error: invalid instruction format

# Invalid immediate
.insn ci  3, 0, a0, 13 # CHECK: <stdin>:[[#@LINE]]:11: error: opcode must be a valid opcode name or an immediate in the range [0, 2]
.insn cr  2, 16, a0, a1 # CHECK: <stdin>:[[#@LINE]]:14: error: immediate must be an integer in the range [0, 15]
.insn ciw 0, 0, a0, 256 # CHECK: <stdin>:[[#@LINE]]:21: error: immediate must be an integer in the range [0, 255]

## Unrecognized opcode name
.insn cr C3, 9, a0, a1 # CHECK: <stdin>:[[#@LINE]]:10: error: opcode must be a valid opcode name or an immediate in the range [0, 2]

## Make fake mnemonics we use to match these in the tablegened asm match table isn't exposed.
.insn_cr  2, 9, a0, a1 # CHECK: <stdin>:[[#@LINE]]:1: error: unknown directive

.insn 0xfffffff0 # CHECK: <stdin>:[[@LINE]]:7: error: encoding value does not fit into instruction
