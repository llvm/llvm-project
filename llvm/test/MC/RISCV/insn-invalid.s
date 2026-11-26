# RUN: llvm-mc --verify -triple riscv32 < %s

# Too many operands
.insn i  0x13,  0,  a0, a1, 13, 14 # expected-error:33 {{unexpected extra operand for instruction}}
.insn r  0x43,  0,  0, fa0, fa1, fa2, fa3, fa4 # expected-error:44 {{unexpected extra operand for instruction}}

# Too few operands
.insn r  0x33,  0,  0, a0, a1 # expected-error:30 {{too few operands for instruction}}
.insn i  0x13,  0,  a0, a1 # expected-error:27 {{too few operands for instruction}}

.insn r  0x33,  0,  0, a0, 13 # expected-error:28 {{invalid operand for instruction}}
.insn i  0x13,  0, a0, a1, a2 # expected-error:28 {{operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]}}

.insn q  0x13,  0,  a0, a1, 13, 14 # expected-error:7 {{invalid instruction format}}

# Invalid immediate
.insn i  0x99,  0, a0, 4(a1) # expected-error:10 {{opcode must be a valid opcode name or an immediate in the range [0, 127]}}
.insn r  0x33,  8,  0, a0, a1, a2 # expected-error:17 {{immediate must be an integer in the range [0, 7]}}
.insn r4 0x43,  0,  4, fa0, fa1, fa2, fa3 # expected-error:21 {{immediate must be an integer in the range [0, 3]}}

# Unrecognized opcode name
.insn r UNKNOWN, 0, a1, a2, a3 # expected-error:9 {{opcode must be a valid opcode name or an immediate in the range [0, 127]}}

# Make fake mnemonics we use to match these in the tablegened asm match table isn't exposed.
.insn_i  0x13,  0,  a0, a1, 13, 14 # expected-error:1 {{unknown directive}}

.insn . # expected-error:7 {{expected absolute expression}}
.insn 0x2, # expected-error:12 {{unknown token in expression}}

.insn 0x4, 0x13, 0 # expected-error:16 {{invalid operand for instruction}}

.insn 0x2, 0xffff # expected-error:7 {{instruction length does not match the encoding}}
.insn 0x2, 0xffffffff # expected-error:7 {{instruction length does not match the encoding}}
.insn 0xffffffffff # expected-error:7 {{encoding value does not fit into instruction}}

.insn 0x0, 0x0 # expected-error:7 {{instruction lengths must be a non-zero multiple of two}}
.insn 0x1, 0xff # expected-error:7 {{instruction lengths must be a non-zero multiple of two}}
.insn 10, 0x000007f # expected-error:7 {{instruction lengths over 64 bits are not supported}}

.insn 0x2, 0x03 # expected-error:7 {{instruction length does not match the encoding}}
.insn 0x2, 0x1f # expected-error:7 {{instruction length does not match the encoding}}
.insn 0x2, 0x3f # expected-error:7 {{instruction length does not match the encoding}}

.insn 0x4, 0x00000001 # expected-error:7 {{instruction length does not match the encoding}}

.insn 0x6, 0x000000000001 # expected-error:7 {{compressed instructions are not allowed}}
.insn 0x8, 0x0000000000000001 # expected-error:7 {{compressed instructions are not allowed}}

.insn 0x2, 0x10001 # expected-error:7 {{encoding value does not fit into instruction}}
.insn 0x4, 0x100000003 # expected-error:7 {{encoding value does not fit into instruction}}
.insn 0x6, 0x100000000001f # expected-error:7 {{encoding value does not fit into instruction}}
.insn 0x8, 0x1000000000000003f # expected-error:12 {{literal value out of range for directive}}

.insn 0x0010 # expected-error:7 {{compressed instructions are not allowed}}
.insn 0x2, 0x0001 # expected-error:7 {{compressed instructions are not allowed}}

.insn 0x2 + 0x2, 0x3 | (31 # expected-error:28 {{expected ')'}}
.insn 0x4 * , 0xbf | (31 << 59) # expected-error:13 {{unknown token in expression}}
