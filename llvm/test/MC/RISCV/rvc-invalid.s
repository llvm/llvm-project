# RUN: not llvm-mc %s -triple=riscv32 2>&1 | FileCheck %s
# RUN: not llvm-mc %s -triple=riscv64 2>&1 | FileCheck %s

reldef:

.global undef

c.addi a0, reldef   # CHECK: :[[@LINE]]:12: error: invalid operand for instruction
c.addi a0, reldef-. # CHECK: :[[@LINE]]:12: error: invalid operand for instruction
c.addi a0, undef    # CHECK: :[[@LINE]]:12: error: invalid operand for instruction
c.addi a0, latedef  # CHECK: :[[@LINE]]:12: error: invalid operand for instruction

c.li a0, reldef   # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
c.li a0, reldef-. # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
c.li a0, undef    # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
c.li a0, latedef  # CHECK: :[[@LINE]]:10: error: invalid operand for instruction

c.andi a0, reldef   # CHECK: :[[@LINE]]:12: error: invalid operand for instruction
c.andi a0, reldef-. # CHECK: :[[@LINE]]:12: error: invalid operand for instruction
c.andi a0, undef    # CHECK: :[[@LINE]]:12: error: invalid operand for instruction
c.andi a0, latedef  # CHECK: :[[@LINE]]:12: error: invalid operand for instruction

.set latedef, 1
