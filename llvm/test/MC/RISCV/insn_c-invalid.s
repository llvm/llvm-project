# RUN: llvm-mc --verify -triple riscv32 -mattr=+c < %s

# Too many operands
.insn ci  1, 0, a0, 13, 14 # expected-error:25 {{unexpected extra operand for instruction}}
.insn cr  2, 9, a0, a1, a2 # expected-error:25 {{unexpected extra operand for instruction}}

## Too few operands
.insn ci  1, 0, a0 # expected-error:19 {{too few operands for instruction}}
.insn cr  2, 9, a0 # expected-error:19 {{too few operands for instruction}}

.insn cr  2, 9, a0, 13 # expected-error:21 {{invalid operand for instruction}}
.insn ci  1, 0, a0, a1 # expected-error:21 {{immediate must be an integer in the range [-32, 31]}}

.insn cq  0x13,  0,  a0, a1, 13, 14 # expected-error:7 {{invalid instruction format}}

# Invalid immediate
.insn ci  3, 0, a0, 13 # expected-error:11 {{opcode must be a valid opcode name or an immediate in the range [0, 2]}}
.insn cr  2, 16, a0, a1 # expected-error:14 {{immediate must be an integer in the range [0, 15]}}
.insn ciw 0, 0, a0, 256 # expected-error:21 {{immediate must be an integer in the range [0, 255]}}

## Unrecognized opcode name
.insn cr C3, 9, a0, a1 # expected-error:10 {{opcode must be a valid opcode name or an immediate in the range [0, 2]}}

## Make fake mnemonics we use to match these in the tablegened asm match table isn't exposed.
.insn_cr  2, 9, a0, a1 # expected-error:1 {{unknown directive}}

.insn 0xfffffff0 # expected-error:7 {{encoding value does not fit into instruction}}
