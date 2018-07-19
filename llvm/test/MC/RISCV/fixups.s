# RUN: llvm-mc -triple riscv32 -riscv-no-aliases < %s -show-encoding \
# RUN:     | FileCheck -check-prefix=CHECK-FIXUP %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump -riscv-no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INSTR %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 %s \
# RUN:     | llvm-readobj -r | FileCheck %s -check-prefix=CHECK-REL

# Checks that fixups that can be resolved within the same object file are
# applied correctly

.LBB0:
lui t1, %hi(val)
# CHECK-FIXUP: fixup A - offset: 0, value: %hi(val), kind: fixup_riscv_hi20
# CHECK-INSTR: lui t1, 74565

lw a0, %lo(val)(t1)
# CHECK-FIXUP: fixup A - offset: 0, value: %lo(val), kind: fixup_riscv_lo12_i
# CHECK-INSTR: lw a0, 1656(t1)
addi a1, t1, %lo(val)
# CHECK-FIXUP: fixup A - offset: 0, value: %lo(val), kind: fixup_riscv_lo12_i
# CHECK-INSTR: addi a1, t1, 1656
sw a0, %lo(val)(t1)
# CHECK-FIXUP: fixup A - offset: 0, value: %lo(val), kind: fixup_riscv_lo12_s
# CHECK-INSTR: sw a0, 1656(t1)

jal zero, .LBB0
# CHECK-FIXUP: fixup A - offset: 0, value: .LBB0, kind: fixup_riscv_jal
# CHECK-INSTR: jal zero, -16
jal zero, .LBB2
# CHECK-FIXUP: fixup A - offset: 0, value: .LBB2, kind: fixup_riscv_jal
# CHECK-INSTR: jal zero, 330996
beq a0, a1, .LBB0
# CHECK-FIXUP: fixup A - offset: 0, value: .LBB0, kind: fixup_riscv_branch
# CHECK-INSTR: beq a0, a1, -24
blt a0, a1, .LBB1
# CHECK-FIXUP: fixup A - offset: 0, value: .LBB1, kind: fixup_riscv_branch
# CHECK-INSTR: blt a0, a1, 1108

.fill 1104

.LBB1:

.fill 329876
addi zero, zero, 0
.LBB2:

.set val, 0x12345678

# CHECK-REL-NOT: R_RISCV
