# RUN: not llvm-mc -triple riscv32 -mattr=+e < %s 2>&1 | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 < %s \
# RUN:     | llvm-objdump --mattr=+e -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefix=CHECK-DIS %s
# RUN: not llvm-mc -triple riscv64 -mattr=+e < %s 2>&1 | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 < %s \
# RUN:     | llvm-objdump --mattr=+e -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefix=CHECK-DIS %s

# Perform a simple check that registers x16-x31 (and the equivalent ABI names)
# are rejected for RV32E/RV64E, when both assembling and disassembling.


# CHECK-DIS: 00001837 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x16, 1
# CHECK-DIS: 000028b7 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x17, 2
# CHECK-DIS: 00003937 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x18, 3
# CHECK-DIS: 000049b7 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x19, 4
# CHECK-DIS: 00005a37 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x20, 5
# CHECK-DIS: 00006ab7 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x21, 6
# CHECK-DIS: 00007b37 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x22, 7
# CHECK-DIS: 00008bb7 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x23, 8
# CHECK-DIS: 00009c37 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x24, 9
# CHECK-DIS: 0000acb7 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x25, 10
# CHECK-DIS: 0000bd37 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x26, 11
# CHECK-DIS: 0000cdb7 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x27, 12
# CHECK-DIS: 0000de37 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x28, 13
# CHECK-DIS: 0000eeb7 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x29, 14
# CHECK-DIS: 0000ff37 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x30, 15
# CHECK-DIS: 00010fb7 <unknown>
# CHECK: :[[@LINE+1]]:5: error: invalid operand for instruction
lui x31, 16

# CHECK-DIS: 00011817 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc a6, 17
# CHECK-DIS: 00012897 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc a7, 18
# CHECK-DIS: 00013917 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc s2, 19
# CHECK-DIS: 00014997 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc s3, 20
# CHECK-DIS: 00015a17 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc s4, 21
# CHECK-DIS: 00016a97 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc s5, 22
# CHECK-DIS: 00017b17 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc s6, 23
# CHECK-DIS: 00018b97 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc s7, 24
# CHECK-DIS: 00019c17 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc s8, 25
# CHECK-DIS: 0001ac97 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc s9, 26
# CHECK-DIS: 0001bd17 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc s10, 27
# CHECK-DIS: 0001cd97 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc s11, 28
# CHECK-DIS: 0001de17 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc t3, 29
# CHECK-DIS: 0001ee97 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc t4, 30
# CHECK-DIS: 0001ff17 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc t5, 31
# CHECK-DIS: 00020f97 <unknown>
# CHECK: :[[@LINE+1]]:7: error: invalid operand for instruction
auipc t6, 32
