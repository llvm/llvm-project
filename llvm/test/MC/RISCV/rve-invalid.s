# RUN: not llvm-mc -triple riscv32 -mattr=+e,+zca < %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple riscv64 -mattr=+e,+zca < %s 2>&1 | FileCheck %s

# Perform a simple check that registers x16-x31 (and the equivalent ABI names)
# are rejected for RV32E/RV64E during assembly.

.option push
.option exact
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x16, 1
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x17, 2
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x18, 3
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x19, 4
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x20, 5
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x21, 6
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x22, 7
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x23, 8
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x24, 9
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x25, 10
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x26, 11
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x27, 12
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x28, 13
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x29, 14
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x30, 15
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x31, 16

# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc a6, 17
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc a7, 18
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc s2, 19
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc s3, 20
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc s4, 21
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc s5, 22
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc s6, 23
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc s7, 24
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc s8, 25
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc s9, 26
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc s10, 27
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc s11, 28
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc t3, 29
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc t4, 30
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc t5, 31
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc t6, 32
.option pop

# CHECK: :[[@LINE+1]]:8: error: register must be a GPR excluding zero (x0)
c.addi x31, 0
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
c.add x16, x17
# CHECK: :[[@LINE+1]]:10: error: register must be a GPR excluding zero (x0)
c.mv x0, x17
