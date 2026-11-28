# RUN: llvm-mc --verify %s -triple=riscv32 -M no-aliases -show-encoding

.insn qc.eai 128, 0, 0, x0, 0 # expected-error:14 {{opcode must be a valid opcode name or an immediate in the range [0, 127]}}

.insn qc.eai 127, 8, 0, x0, 0 # expected-error:19 {{immediate must be an integer in the range [0, 7]}}

.insn qc.eai 127, 7, 2, x0, 0 # expected-error:22 {{immediate must be an integer in the range [0, 1]}}

.insn qc.eai 127, 7, 1, not_a_reg, 0 # expected-error:25 {{invalid operand for instruction}}

.insn qc.eai 127, 7, 1, x31, 0x100000000 # expected-error:30 {{immediate must be an integer in the range [-2147483648, 4294967295]}}

.insn qc.eai 126, 7, 1, x31, 0xFFFFFFFF, extra # expected-error:42 {{unexpected extra operand for instruction}}

.insn qc.ei 128, 0, 0, x31, x0, 0 # expected-error:13 {{opcode must be a valid opcode name or an immediate in the range [0, 127]}}

.insn qc.ei 127, 8, 0, x0, x0, 0 # expected-error:18 {{immediate must be an integer in the range [0, 7]}}

.insn qc.ei 127, 7, 4, x0, x0, 0 # expected-error:21 {{immediate must be an integer in the range [0, 3]}}

.insn qc.ei 127, 7, 3, not_a_reg, x0, 0 # expected-error:24 {{invalid operand for instruction}}

.insn qc.ei 127, 7, 3, x31, not_a_reg, 0 # expected-error:29 {{invalid operand for instruction}}

.insn qc.ei 127, 7, 3, x31, x31, 0x2000000 # expected-error:34 {{immediate must be an integer in the range [-33554432, 33554431]}}

.insn qc.ei 127, 7, 3, x31, x31, 0x1000000, extra # expected-error:45 {{unexpected extra operand for instruction}}

.insn qc.ei 126, 7, 3, x31, 0x2000000(x0) # expected-error:29 {{immediate must be an integer in the range [-33554432, 33554431]}}

.insn qc.ei 126, 7, 3, x31, 0x1000000(not_a_reg) # expected-error:39 {{expected register}}

.insn qc.ei 126, 7, 3, x31, 0x1000000(x31), extra # expected-error:45 {{unexpected extra operand for instruction}}

.insn qc.eb 128, 0, 0, x0, 0, 0 # expected-error:13 {{opcode must be a valid opcode name or an immediate in the range [0, 127]}}

.insn qc.eb 127, 8, 0, x0, 0, 0 # expected-error:18 {{immediate must be an integer in the range [0, 7]}}

.insn qc.eb 127, 7, 32, x0, 0, 0 # expected-error:21 {{immediate must be an integer in the range [0, 31]}}

.insn qc.eb 127, 7, 31, not_a_reg, 0, 0 # expected-error:25 {{invalid operand for instruction}}

.insn qc.eb 127, 7, 31, x31, 0x8000, 0 # expected-error:30 {{immediate must be an integer in the range [-32768, 32767]}}

.insn qc.eb 127, 7, 31, x31, 0x4000, 0x1000 # expected-error:38 {{immediate must be a multiple of 2 bytes in the range [-4096, 4094]}}

.insn qc.eb 127, 7, 31, x31, 0x4000, 0x800, extra # expected-error:45 {{unexpected extra operand for instruction}}


.insn qc.ej 128, 0, 0, 0, 0 # expected-error:13 {{opcode must be a valid opcode name or an immediate in the range [0, 127]}}

.insn qc.ej 127, 8, 0, 0, 0 # expected-error:18 {{immediate must be an integer in the range [0, 7]}}

.insn qc.ej 127, 7, 4, 0, 0 # expected-error:21 {{immediate must be an integer in the range [0, 3]}}

.insn qc.ej 127, 7, 3, 32, 0 # expected-error:24 {{immediate must be an integer in the range [0, 31]}}

.insn qc.ej 127, 7, 3, 31, 0x100000000 # expected-error:28 {{operand must be a multiple of 2 bytes in the range [-2147483648, 2147483646]}}

.insn qc.ej 127, 7, 3, 31, 0x80000000, extra # expected-error:40 {{unexpected extra operand for instruction}}

.insn qc.es 128, 0, 0, x0, 0(x0) # expected-error:13 {{opcode must be a valid opcode name or an immediate in the range [0, 127]}}

.insn qc.es 127, 8, 0, x0, 0(x0) # expected-error:18 {{immediate must be an integer in the range [0, 7]}}

.insn qc.es 127, 7, 4, x0, 0(x0) # expected-error:21 {{immediate must be an integer in the range [0, 3]}}

.insn qc.es 127, 7, 3, not_a_reg, 0(x0) # expected-error:24 {{invalid operand for instruction}}

.insn qc.es 127, 7, 3, x31, 0x2000000(x0) # expected-error:29 {{immediate must be an integer in the range [-33554432, 33554431]}}

.insn qc.es 127, 7, 3, x31, 0x1000000(not_a_reg) # expected-error:39 {{expected register}}

.insn qc.es 127, 7, 3, x31, 0x1000000(x31), extra # expected-error:45 {{unexpected extra operand for instruction}}
