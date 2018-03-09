# RUN: llvm-mc %s -triple riscv32 -mattr=+c -show-encoding \
# RUN:     | FileCheck -check-prefix=CHECK-FIXUP %s
# RUN: llvm-mc -triple riscv32 -filetype=obj -mattr=+c < %s \
# RUN:     | llvm-objdump -mattr=+c -d - | FileCheck -check-prefix=CHECK-INSTR %s

.LBB0_2:
# CHECK-FIXUP:   fixup A - offset: 0, value: .LBB0_2, kind: fixup_riscv_rvc_jump
# CHECK-INSTR: c.j     0
c.j     .LBB0_2
# CHECK:   fixup A - offset: 0, value: func1, kind: fixup_riscv_rvc_jump
# CHECK-INSTR: c.jal   0
c.jal   func1
# CHECK-FIXUP:   fixup A - offset: 0, value: .LBB0_2, kind: fixup_riscv_rvc_branch
# CHECK-INSTR: c.beqz  a3, -4
c.beqz  a3, .LBB0_2
# CHECK-FIXUP:   fixup A - offset: 0, value: .LBB0_2, kind: fixup_riscv_rvc_branch
# CHECK-INSTR: c.bnez  a5, -6
c.bnez  a5, .LBB0_2
