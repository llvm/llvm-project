# RUN: not llvm-mc %s -triple=riscv32 -M no-aliases -show-encoding \
# RUN:     2>&1 | FileCheck -check-prefixes=CHECK-ERR %s

.insn qc.eai 128, 0, 0, x0, 0
# CHECK-ERR: [[@LINE-1]]:14: error: opcode must be a valid opcode name or an immediate in the range [0, 127]

.insn qc.eai 127, 8, 0, x0, 0
# CHECK-ERR: [[@LINE-1]]:19: error: immediate must be an integer in the range [0, 7]

.insn qc.eai 127, 7, 2, x0, 0
# CHECK-ERR: [[@LINE-1]]:22: error: immediate must be an integer in the range [0, 1]

.insn qc.eai 127, 7, 1, not_a_reg, 0
# CHECK-ERR: [[@LINE-1]]:25: error: invalid operand for instruction

.insn qc.eai 127, 7, 1, x31, 0x100000000
# CHECK-ERR: [[@LINE-1]]:30: error: immediate must be an integer in the range [-2147483648, 4294967295]

.insn qc.eai 126, 7, 1, x31, 0xFFFFFFFF, extra
# CHECK-ERR: [[@LINE-1]]:42: error: invalid operand for instruction

.insn qc.ei 128, 0, 0, x31, x0, 0
# CHECK-ERR: [[@LINE-1]]:13: error: opcode must be a valid opcode name or an immediate in the range [0, 127]

.insn qc.ei 127, 8, 0, x0, x0, 0
# CHECK-ERR: [[@LINE-1]]:18: error: immediate must be an integer in the range [0, 7]

.insn qc.ei 127, 7, 4, x0, x0, 0
# CHECK-ERR: [[@LINE-1]]:21: error: immediate must be an integer in the range [0, 3]

.insn qc.ei 127, 7, 3, not_a_reg, x0, 0
# CHECK-ERR: [[@LINE-1]]:24: error: invalid operand for instruction

.insn qc.ei 127, 7, 3, x31, not_a_reg, 0
# CHECK-ERR: [[@LINE-1]]:29: error: immediate must be an integer in the range [-33554432, 33554431]

.insn qc.ei 127, 7, 3, x31, x31, 0x2000000
# CHECK-ERR: [[@LINE-1]]:34: error: immediate must be an integer in the range [-33554432, 33554431]

.insn qc.ei 127, 7, 3, x31, x31, 0x1000000, extra
# CHECK-ERR: [[@LINE-1]]:45: error: invalid operand for instruction

.insn qc.ei 126, 7, 3, x31, 0x2000000(x0)
# CHECK-ERR: [[@LINE-1]]:29: error: immediate must be an integer in the range [-33554432, 33554431]

.insn qc.ei 126, 7, 3, x31, 0x1000000(not_a_reg)
# CHECK-ERR: [[@LINE-1]]:39: error: expected register

.insn qc.ei 126, 7, 3, x31, 0x1000000(x31), extra
# CHECK-ERR: [[@LINE-1]]:45: error: invalid operand for instruction

.insn qc.eb 128, 0, 0, x0, 0, 0
# CHECK-ERR: [[@LINE-1]]:13: error: opcode must be a valid opcode name or an immediate in the range [0, 127]

.insn qc.eb 127, 8, 0, x0, 0, 0
# CHECK-ERR: [[@LINE-1]]:18: error: immediate must be an integer in the range [0, 7]

.insn qc.eb 127, 7, 32, x0, 0, 0
# CHECK-ERR: [[@LINE-1]]:21: error: immediate must be an integer in the range [0, 31]

.insn qc.eb 127, 7, 31, not_a_reg, 0, 0
# CHECK-ERR: [[@LINE-1]]:25: error: invalid operand for instruction

.insn qc.eb 127, 7, 31, x31, 0x8000, 0
# CHECK-ERR: [[@LINE-1]]:30: error: immediate must be an integer in the range [-32768, 32767]

.insn qc.eb 127, 7, 31, x31, 0x4000, 0x1000
# CHECK-ERR: [[@LINE-1]]:38: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]

.insn qc.eb 127, 7, 31, x31, 0x4000, 0x800, extra
# CHECK-ERR: [[@LINE-1]]:45: error: invalid operand for instruction


.insn qc.ej 128, 0, 0, 0, 0
# CHECK-ERR: [[@LINE-1]]:13: error: opcode must be a valid opcode name or an immediate in the range [0, 127]

.insn qc.ej 127, 8, 0, 0, 0
# CHECK-ERR: [[@LINE-1]]:18: error: immediate must be an integer in the range [0, 7]

.insn qc.ej 127, 7, 4, 0, 0
# CHECK-ERR: [[@LINE-1]]:21: error: immediate must be an integer in the range [0, 3]

.insn qc.ej 127, 7, 3, 32, 0
# CHECK-ERR: [[@LINE-1]]:24: error: immediate must be an integer in the range [0, 31]

.insn qc.ej 127, 7, 3, 31, 0x100000000
# CHECK-ERR: [[@LINE-1]]:28: error: operand must be a multiple of 2 bytes in the range [-2147483648, 2147483646]

.insn qc.ej 127, 7, 3, 31, 0x80000000, extra
# CHECK-ERR: [[@LINE-1]]:40: error: invalid operand for instruction

.insn qc.es 128, 0, 0, x0, 0(x0)
# CHECK-ERR: [[@LINE-1]]:13: error: opcode must be a valid opcode name or an immediate in the range [0, 127]

.insn qc.es 127, 8, 0, x0, 0(x0)
# CHECK-ERR: [[@LINE-1]]:18: error: immediate must be an integer in the range [0, 7]

.insn qc.es 127, 7, 4, x0, 0(x0)
# CHECK-ERR: [[@LINE-1]]:21: error: immediate must be an integer in the range [0, 3]

.insn qc.es 127, 7, 3, not_a_reg, 0(x0)
# CHECK-ERR: [[@LINE-1]]:24: error: invalid operand for instruction

.insn qc.es 127, 7, 3, x31, 0x2000000(x0)
# CHECK-ERR: [[@LINE-1]]:29: error: immediate must be an integer in the range [-33554432, 33554431]

.insn qc.es 127, 7, 3, x31, 0x1000000(not_a_reg)
# CHECK-ERR: [[@LINE-1]]:39: error: expected register

.insn qc.es 127, 7, 3, x31, 0x1000000(x31), extra
# CHECK-ERR: [[@LINE-1]]:45: error: invalid operand for instruction
