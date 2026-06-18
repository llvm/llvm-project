# RUN: llvm-mc -triple riscv32 -filetype=obj -mattr=+c < %s \
# RUN:     | llvm-objdump -d -M no-aliases - | FileCheck --check-prefix=CHECK-INSTR %s
# RUN: llvm-mc -filetype=obj -mattr=+c -triple=riscv32 %s \
# RUN:     | llvm-readobj -r - | FileCheck %s -check-prefix=CHECK-REL

.LBB0_2:
# CHECK-INSTR: c.j     0
c.j     .LBB0_2
# CHECK-INSTR: c.jal   0x8
c.jal   func1
# CHECK-INSTR: c.beqz  a3, 0x0
c.beqz  a3, .LBB0_2
# CHECK-INSTR: c.bnez  a5, 0x0
c.bnez  a5, .LBB0_2

func1:
  nop

# CHECK-REL-NOT: R_RISCV
