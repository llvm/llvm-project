# RUN: not llvm-mc -triple=riscv32 -mattr=+c < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-RV32 %s
# RUN: not llvm-mc -triple=riscv64 -mattr=+c < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-RV64 %s

c.nop 0 # CHECK: :[[@LINE]]:7: error: immediate must be non-zero in the range [-32, 31]

c.addi x0, 33 # CHECK: :[[@LINE]]:12: error: immediate must be non-zero in the range [-32, 31]

c.li x0, 42 # CHECK: :[[@LINE]]:10: error: immediate must be an integer in the range [-32, 31]

c.lui x0, 0 # CHECK: :[[@LINE]]:11: error: immediate must be in [0xfffe0, 0xfffff] or [1, 31]

c.mv x0, x0 # CHECK: :[[@LINE]]:10: error: invalid operand for instruction

c.add x0, x0 # CHECK: :[[@LINE]]:11: error: invalid operand for instruction

c.slli x0, 0 # CHECK-RV32: :[[@LINE]]:12: error: immediate must be an integer in the range [1, 31]
c.slli x0, 32 # CHECK-RV32: :[[@LINE]]:12: error: immediate must be an integer in the range [1, 31]

c.slli x0, 0 # CHECK-RV64: :[[@LINE]]:12: error: immediate must be an integer in the range [1, 63]

c.srli64 x30 # CHECK: :[[@LINE]]:10: error: invalid operand for instruction

c.srai64 x31 # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
