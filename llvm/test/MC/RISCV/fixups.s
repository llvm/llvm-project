# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump --no-print-imm-hex -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INSTR %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 %s \
# RUN:     | llvm-readobj -r - | FileCheck %s -check-prefix=CHECK-REL

# Checks that fixups that can be resolved within the same object file are
# applied correctly

.LBB0:
lui t1, %hi(val)
# CHECK-INSTR: lui t1, 74565

lw a0, %lo(val)(t1)
# CHECK-INSTR: lw a0, 1656(t1)
addi a1, t1, %lo(val)
# CHECK-INSTR: addi a1, t1, 1656
sw a0, %lo(val)(t1)
# CHECK-INSTR: sw a0, 1656(t1)

1:
auipc t1, %pcrel_hi(.LBB0)
# CHECK-INSTR: auipc t1, 0
addi t1, t1, %pcrel_lo(1b)
# CHECK-INSTR: addi t1, t1, -16
sw t1, %pcrel_lo(1b)(t1)
# CHECK-INSTR: sw t1, -16(t1)

jal zero, .LBB0
# CHECK-INSTR: jal zero, 0x0
jal zero, .LBB2
# CHECK-INSTR: jal zero, 0x50d14
beq a0, a1, .LBB0
# CHECK-INSTR: beq a0, a1, 0x0
blt a0, a1, .LBB1
# CHECK-INSTR: blt a0, a1, 0x47c

.fill 1104

.LBB1:

.fill 329876
addi zero, zero, 0
.LBB2:

.set val, 0x12345678

# CHECK-REL-NOT: R_RISCV

# Testing the function call offset could resolved by assembler
# when the function and the callsite within the same compile unit
# and the linker relaxation is disabled.
func:
.fill 100
call func
# CHECK-INSTR: auipc   ra, 0
# CHECK-INSTR: jalr    ra, -100(ra)

.fill 10000
call func
# CHECK-INSTR: auipc   ra, 1048574
# CHECK-INSTR: jalr    ra, -1916(ra)

.fill 20888
call func
# CHECK-INSTR: auipc   ra, 1048568
# CHECK-INSTR: jalr    ra, 1764(ra)
