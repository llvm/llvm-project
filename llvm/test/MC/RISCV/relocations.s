# RUN: llvm-mc -triple riscv32 -riscv-no-aliases < %s -show-encoding \
# RUN:     | FileCheck -check-prefix=INSTR -check-prefix=FIXUP %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+c < %s \
# RUN:     | llvm-readobj -r | FileCheck -check-prefix=RELOC %s

# Check prefixes:
# RELOC - Check the relocation in the object.
# FIXUP - Check the fixup on the instruction.
# INSTR - Check the instruction is handled properly by the ASMPrinter

.long foo
# RELOC: R_RISCV_32 foo

.quad foo
# RELOC: R_RISCV_64 foo

lui t1, %hi(foo)
# RELOC: R_RISCV_HI20 foo 0x0
# INSTR: lui t1, %hi(foo)
# FIXUP: fixup A - offset: 0, value: %hi(foo), kind: fixup_riscv_hi20

lui t1, %hi(foo+4)
# RELOC: R_RISCV_HI20 foo 0x4
# INSTR: lui t1, %hi(foo+4)
# FIXUP: fixup A - offset: 0, value: %hi(foo+4), kind: fixup_riscv_hi20

addi t1, t1, %lo(foo)
# RELOC: R_RISCV_LO12_I foo 0x0
# INSTR: addi t1, t1, %lo(foo)
# FIXUP: fixup A - offset: 0, value: %lo(foo), kind: fixup_riscv_lo12_i

addi t1, t1, %lo(foo+4)
# RELOC: R_RISCV_LO12_I foo 0x4
# INSTR: addi t1, t1, %lo(foo+4)
# FIXUP: fixup A - offset: 0, value: %lo(foo+4), kind: fixup_riscv_lo12_i

sb t1, %lo(foo)(a2)
# RELOC: R_RISCV_LO12_S foo 0x0
# INSTR: sb t1, %lo(foo)(a2)
# FIXUP: fixup A - offset: 0, value: %lo(foo), kind: fixup_riscv_lo12_s

sb t1, %lo(foo+4)(a2)
# RELOC: R_RISCV_LO12_S foo 0x4
# INSTR: sb t1, %lo(foo+4)(a2)
# FIXUP: fixup A - offset: 0, value: %lo(foo+4), kind: fixup_riscv_lo12_s

.L0:
auipc t1, %pcrel_hi(foo)
# RELOC: R_RISCV_PCREL_HI20 foo 0x0
# INSTR: auipc t1, %pcrel_hi(foo)
# FIXUP: fixup A - offset: 0, value: %pcrel_hi(foo), kind: fixup_riscv_pcrel_hi20

auipc t1, %pcrel_hi(foo+4)
# RELOC: R_RISCV_PCREL_HI20 foo 0x4
# INSTR: auipc t1, %pcrel_hi(foo+4)
# FIXUP: fixup A - offset: 0, value: %pcrel_hi(foo+4), kind: fixup_riscv_pcrel_hi20

addi t1, t1, %pcrel_lo(.L0)
# RELOC: R_RISCV_PCREL_LO12_I .L0 0x0
# INSTR: addi t1, t1, %pcrel_lo(.L0)
# FIXUP: fixup A - offset: 0, value: %pcrel_lo(.L0), kind: fixup_riscv_pcrel_lo12_i

sb t1, %pcrel_lo(.L0)(a2)
# RELOC: R_RISCV_PCREL_LO12_S .L0 0x0
# INSTR: sb t1, %pcrel_lo(.L0)(a2)
# FIXUP: fixup A - offset: 0, value: %pcrel_lo(.L0), kind: fixup_riscv_pcrel_lo12_s

jal zero, foo
# RELOC: R_RISCV_JAL
# INSTR: jal zero, foo
# FIXUP: fixup A - offset: 0, value: foo, kind: fixup_riscv_jal

bgeu a0, a1, foo
# RELOC: R_RISCV_BRANCH
# INSTR: bgeu a0, a1, foo
# FIXUP: fixup A - offset: 0, value: foo, kind: fixup_riscv_branch
