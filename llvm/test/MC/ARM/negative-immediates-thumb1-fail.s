# RUN: not llvm-mc -triple thumbv7 -mcpu=cortex-m0 %s 2>&1 | FileCheck %s

.thumb

ADDs r1, r0, #0xFFFFFFF5
# CHECK: error: invalid instruction, any one of the following would fix this:
# CHECK-DAG: note: instruction requires: thumb2
# CHECK-DAG: note: invalid operand for instruction
# CHECK-DAG: note: operand must be an immediate in the range [0,7]
# CHECK-DAG: note: operand must be a register in range [r0, r7]

ADDs r0, #0xFFFFFEFF
# CHECK: error: invalid instruction, any one of the following would fix this:
# CHECK-DAG: note: invalid operand for instruction
# CHECK-DAG: note: operand must be an immediate in the range [0,255]

SUBs r1, r0, #0xFFFFFFF5
# CHECK: error: invalid instruction, any one of the following would fix this:
# CHECK-DAG: note: invalid operand for instruction
# CHECK-DAG: note: operand must be an immediate in the range [0,7]
# CHECK-DAG: note: operand must be a register in range [r0, r7]

SUBs r0, #0xFFFFFEFF
# CHECK: error: invalid instruction, any one of the following would fix this:
# CHECK-DAG: note: invalid operand for instruction
# CHECK-DAG: note: operand must be an immediate in the range [0,255]

ORRs r0, r1, #0xFFFFFF00
# CHECK: error: invalid instruction, any one of the following would fix this:
# CHECK-DAG: note: instruction requires: thumb2
# CHECK-DAG: note: too many operands for instruction

ORNs r0, r1, #0xFFFFFF00
# CHECK: error: instruction requires: thumb2
