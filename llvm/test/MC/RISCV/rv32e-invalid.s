# RUN: not llvm-mc -triple riscv32 -mattr=+e < %s 2>&1 | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 < %s \
# RUN:     | llvm-objdump --mattr=+e -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefix=CHECK-DIS %s

# Perform a simple sanity check that registers x16-x31 (and the equivalent
# ABI names) are rejected for RV32E, when both assembling and disassembling.


# CHECK-DIS: 37 18 00 00 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x16, 1
# CHECK-DIS: b7 28 00 00 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x17, 2
# CHECK-DIS: 37 39 00 00 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x18, 3
# CHECK-DIS: b7 49 00 00 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x19, 4
# CHECK-DIS: 37 5a 00 00 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x20, 5
# CHECK-DIS: b7 6a 00 00 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x21, 6
# CHECK-DIS: 37 7b 00 00 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x22, 7
# CHECK-DIS: b7 8b 00 00 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x23, 8
# CHECK-DIS: 37 9c 00 00 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x24, 9
# CHECK-DIS: b7 ac 00 00 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x25, 10
# CHECK-DIS: 37 bd 00 00 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x26, 11
# CHECK-DIS: b7 cd 00 00 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x27, 12
# CHECK-DIS: 37 de 00 00 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x28, 13
# CHECK-DIS: b7 ee 00 00 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x29, 14
# CHECK-DIS: 37 ff 00 00 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x30, 15
# CHECK-DIS: b7 0f 01 00 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x31, 16

# CHECK-DIS: 17 18 01 00 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc a6, 17
# CHECK-DIS: 97 28 01 00 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc a7, 18
# CHECK-DIS: 17 39 01 00 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc s2, 19
# CHECK-DIS: 97 49 01 00 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc s3, 20
# CHECK-DIS: 17 5a 01 00 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc s4, 21
# CHECK-DIS: 97 6a 01 00 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc s5, 22
# CHECK-DIS: 17 7b 01 00 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc s6, 23
# CHECK-DIS: 97 8b 01 00 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc s7, 24
# CHECK-DIS: 17 9c 01 00 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc s8, 25
# CHECK-DIS: 97 ac 01 00 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc s9, 26
# CHECK-DIS: 17 bd 01 00 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc s10, 27
# CHECK-DIS: 97 cd 01 00 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc s11, 28
# CHECK-DIS: 17 de 01 00 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc t3, 29
# CHECK-DIS: 97 ee 01 00 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc t4, 30
# CHECK-DIS: 17 ff 01 00 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc t5, 31
# CHECK-DIS: 97 0f 02 00 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc t6, 32
