# RUN: llvm-mc --verify %s -triple=riscv32 -M no-aliases -show-encoding

.insn qc.eai 128, 0, 0, x0, 0
# expected-error@-1:14 {{opcode must be a valid opcode name or an immediate in the range [0, 127]}}

.insn qc.eai 127, 8, 0, x0, 0
# expected-error@-1:19 {{immediate must be an integer in the range [0, 7]}}

.insn qc.eai 127, 7, 2, x0, 0
# expected-error@-1:22 {{immediate must be an integer in the range [0, 1]}}

.insn qc.eai 127, 7, 1, not_a_reg, 0
# expected-error@-1:25 {{invalid operand for instruction}}

.insn qc.eai 127, 7, 1, x31, 0x100000000
# expected-error@-1:30 {{immediate must be an integer in the range [-2147483648, 4294967295]}}

.insn qc.eai 126, 7, 1, x31, 0xFFFFFFFF, extra
# expected-error@-1:42 {{unexpected extra operand for instruction}}

.insn qc.ei 128, 0, 0, x31, x0, 0
# expected-error@-1:13 {{opcode must be a valid opcode name or an immediate in the range [0, 127]}}

.insn qc.ei 127, 8, 0, x0, x0, 0
# expected-error@-1:18 {{immediate must be an integer in the range [0, 7]}}

.insn qc.ei 127, 7, 4, x0, x0, 0
# expected-error@-1:21 {{immediate must be an integer in the range [0, 3]}}

.insn qc.ei 127, 7, 3, not_a_reg, x0, 0
# expected-error@-1:24 {{invalid operand for instruction}}

.insn qc.ei 127, 7, 3, x31, not_a_reg, 0
# expected-error@-1:29 {{invalid operand for instruction}}

.insn qc.ei 127, 7, 3, x31, x31, 0x2000000
# expected-error@-1:34 {{immediate must be an integer in the range [-33554432, 33554431]}}

.insn qc.ei 127, 7, 3, x31, x31, 0x1000000, extra
# expected-error@-1:45 {{unexpected extra operand for instruction}}

.insn qc.ei 126, 7, 3, x31, 0x2000000(x0)
# expected-error@-1:29 {{immediate must be an integer in the range [-33554432, 33554431]}}

.insn qc.ei 126, 7, 3, x31, 0x1000000(not_a_reg)
# expected-error@-1:39 {{expected register}}

.insn qc.ei 126, 7, 3, x31, 0x1000000(x31), extra
# expected-error@-1:45 {{unexpected extra operand for instruction}}

.insn qc.eb 128, 0, 0, x0, 0, 0
# expected-error@-1:13 {{opcode must be a valid opcode name or an immediate in the range [0, 127]}}

.insn qc.eb 127, 8, 0, x0, 0, 0
# expected-error@-1:18 {{immediate must be an integer in the range [0, 7]}}

.insn qc.eb 127, 7, 32, x0, 0, 0
# expected-error@-1:21 {{immediate must be an integer in the range [0, 31]}}

.insn qc.eb 127, 7, 31, not_a_reg, 0, 0
# expected-error@-1:25 {{invalid operand for instruction}}

.insn qc.eb 127, 7, 31, x31, 0x8000, 0
# expected-error@-1:30 {{immediate must be an integer in the range [-32768, 32767]}}

.insn qc.eb 127, 7, 31, x31, 0x4000, 0x1000
# expected-error@-1:38 {{immediate must be a multiple of 2 bytes in the range [-4096, 4094]}}

.insn qc.eb 127, 7, 31, x31, 0x4000, 0x800, extra
# expected-error@-1:45 {{unexpected extra operand for instruction}}


.insn qc.ej 128, 0, 0, 0, 0
# expected-error@-1:13 {{opcode must be a valid opcode name or an immediate in the range [0, 127]}}

.insn qc.ej 127, 8, 0, 0, 0
# expected-error@-1:18 {{immediate must be an integer in the range [0, 7]}}

.insn qc.ej 127, 7, 4, 0, 0
# expected-error@-1:21 {{immediate must be an integer in the range [0, 3]}}

.insn qc.ej 127, 7, 3, 32, 0
# expected-error@-1:24 {{immediate must be an integer in the range [0, 31]}}

.insn qc.ej 127, 7, 3, 31, 0x100000000
# expected-error@-1:28 {{operand must be a multiple of 2 bytes in the range [-2147483648, 2147483646]}}

.insn qc.ej 127, 7, 3, 31, 0x80000000, extra
# expected-error@-1:40 {{unexpected extra operand for instruction}}

.insn qc.es 128, 0, 0, x0, 0(x0)
# expected-error@-1:13 {{opcode must be a valid opcode name or an immediate in the range [0, 127]}}

.insn qc.es 127, 8, 0, x0, 0(x0)
# expected-error@-1:18 {{immediate must be an integer in the range [0, 7]}}

.insn qc.es 127, 7, 4, x0, 0(x0)
# expected-error@-1:21 {{immediate must be an integer in the range [0, 3]}}

.insn qc.es 127, 7, 3, not_a_reg, 0(x0)
# expected-error@-1:24 {{invalid operand for instruction}}

.insn qc.es 127, 7, 3, x31, 0x2000000(x0)
# expected-error@-1:29 {{immediate must be an integer in the range [-33554432, 33554431]}}

.insn qc.es 127, 7, 3, x31, 0x1000000(not_a_reg)
# expected-error@-1:39 {{expected register}}

.insn qc.es 127, 7, 3, x31, 0x1000000(x31), extra
# expected-error@-1:45 {{unexpected extra operand for instruction}}
