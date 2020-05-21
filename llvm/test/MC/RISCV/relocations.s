# RUN: llvm-mc -triple riscv32 -riscv-no-aliases < %s -show-encoding \
# RUN:     | FileCheck -check-prefix=INSTR -check-prefix=FIXUP %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+c < %s \
# RUN:     | llvm-readobj -r | FileCheck -check-prefix=RELOC %s

# COM: Check prefixes:
# COM: RELOC - Check the relocation in the object.
# COM: FIXUP - Check the fixup on the instruction.
# COM: INSTR - Check the instruction is handled properly by the ASMPrinter

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

lui t1, %tprel_hi(foo)
# RELOC: R_RISCV_TPREL_HI20 foo 0x0
# INSTR: lui t1, %tprel_hi(foo)
# FIXUP: fixup A - offset: 0, value: %tprel_hi(foo), kind: fixup_riscv_tprel_hi20

lui t1, %tprel_hi(foo+4)
# RELOC: R_RISCV_TPREL_HI20 foo 0x4
# INSTR: lui t1, %tprel_hi(foo+4)
# FIXUP: fixup A - offset: 0, value: %tprel_hi(foo+4), kind: fixup_riscv_tprel_hi20

addi t1, t1, %lo(foo)
# RELOC: R_RISCV_LO12_I foo 0x0
# INSTR: addi t1, t1, %lo(foo)
# FIXUP: fixup A - offset: 0, value: %lo(foo), kind: fixup_riscv_lo12_i

addi t1, t1, %lo(foo+4)
# RELOC: R_RISCV_LO12_I foo 0x4
# INSTR: addi t1, t1, %lo(foo+4)
# FIXUP: fixup A - offset: 0, value: %lo(foo+4), kind: fixup_riscv_lo12_i

addi t1, t1, %tprel_lo(foo)
# RELOC: R_RISCV_TPREL_LO12_I foo 0x0
# INSTR: addi t1, t1, %tprel_lo(foo)
# FIXUP: fixup A - offset: 0, value: %tprel_lo(foo), kind: fixup_riscv_tprel_lo12_i

addi t1, t1, %tprel_lo(foo+4)
# RELOC: R_RISCV_TPREL_LO12_I foo 0x4
# INSTR: addi t1, t1, %tprel_lo(foo+4)
# FIXUP: fixup A - offset: 0, value: %tprel_lo(foo+4), kind: fixup_riscv_tprel_lo12_i

sb t1, %lo(foo)(a2)
# RELOC: R_RISCV_LO12_S foo 0x0
# INSTR: sb t1, %lo(foo)(a2)
# FIXUP: fixup A - offset: 0, value: %lo(foo), kind: fixup_riscv_lo12_s

sb t1, %lo(foo+4)(a2)
# RELOC: R_RISCV_LO12_S foo 0x4
# INSTR: sb t1, %lo(foo+4)(a2)
# FIXUP: fixup A - offset: 0, value: %lo(foo+4), kind: fixup_riscv_lo12_s

sb t1, %tprel_lo(foo)(a2)
# RELOC: R_RISCV_TPREL_LO12_S foo 0x0
# INSTR: sb t1, %tprel_lo(foo)(a2)
# FIXUP: fixup A - offset: 0, value: %tprel_lo(foo), kind: fixup_riscv_tprel_lo12_s

sb t1, %tprel_lo(foo+4)(a2)
# RELOC: R_RISCV_TPREL_LO12_S foo 0x4
# INSTR: sb t1, %tprel_lo(foo+4)(a2)
# FIXUP: fixup A - offset: 0, value: %tprel_lo(foo+4), kind: fixup_riscv_tprel_lo12_s

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

.L1:
auipc t1, %got_pcrel_hi(foo)
# RELOC: R_RISCV_GOT_HI20 foo 0x0
# INSTR: auipc t1, %got_pcrel_hi(foo)
# FIXUP: fixup A - offset: 0, value: %got_pcrel_hi(foo), kind: fixup_riscv_got_hi20

addi t1, t1, %pcrel_lo(.L1)
# RELOC: R_RISCV_PCREL_LO12_I .L1 0x0
# INSTR: addi t1, t1, %pcrel_lo(.L1)
# FIXUP: fixup A - offset: 0, value: %pcrel_lo(.L1), kind: fixup_riscv_pcrel_lo12_i

sb t1, %pcrel_lo(.L1)(a2)
# RELOC: R_RISCV_PCREL_LO12_S .L1 0x0
# INSTR: sb t1, %pcrel_lo(.L1)(a2)
# FIXUP: fixup A - offset: 0, value: %pcrel_lo(.L1), kind: fixup_riscv_pcrel_lo12_s

# Check that GOT relocations aren't evaluated to a constant when the symbol is
# in the same object file.
.L2:
auipc t1, %got_pcrel_hi(.L1)
# RELOC: R_RISCV_GOT_HI20 .L1 0x0
# INSTR: auipc t1, %got_pcrel_hi(.L1)
# FIXUP: fixup A - offset: 0, value: %got_pcrel_hi(.L1), kind: fixup_riscv_got_hi20

addi t1, t1, %pcrel_lo(.L2)
# RELOC: R_RISCV_PCREL_LO12_I .L2 0x0
# INSTR: addi t1, t1, %pcrel_lo(.L2)
# FIXUP: fixup A - offset: 0, value: %pcrel_lo(.L2), kind: fixup_riscv_pcrel_lo12_i

sb t1, %pcrel_lo(.L2)(a2)
# RELOC: R_RISCV_PCREL_LO12_S .L2 0x0
# INSTR: sb t1, %pcrel_lo(.L2)(a2)
# FIXUP: fixup A - offset: 0, value: %pcrel_lo(.L2), kind: fixup_riscv_pcrel_lo12_s

.L3:
auipc t1, %tls_ie_pcrel_hi(foo)
# RELOC: R_RISCV_TLS_GOT_HI20 foo 0x0
# INSTR: auipc t1, %tls_ie_pcrel_hi(foo)
# FIXUP: fixup A - offset: 0, value: %tls_ie_pcrel_hi(foo), kind: fixup_riscv_tls_got_hi20

addi t1, t1, %pcrel_lo(.L3)
# RELOC: R_RISCV_PCREL_LO12_I .L3 0x0
# INSTR: addi t1, t1, %pcrel_lo(.L3)
# FIXUP: fixup A - offset: 0, value: %pcrel_lo(.L3), kind: fixup_riscv_pcrel_lo12_i

sb t1, %pcrel_lo(.L3)(a2)
# RELOC: R_RISCV_PCREL_LO12_S .L3 0x0
# INSTR: sb t1, %pcrel_lo(.L3)(a2)
# FIXUP: fixup A - offset: 0, value: %pcrel_lo(.L3), kind: fixup_riscv_pcrel_lo12_s

.L4:
auipc t1, %tls_gd_pcrel_hi(foo)
# RELOC: R_RISCV_TLS_GD_HI20 foo 0x0
# INSTR: auipc t1, %tls_gd_pcrel_hi(foo)
# FIXUP: fixup A - offset: 0, value: %tls_gd_pcrel_hi(foo), kind: fixup_riscv_tls_gd_hi20

addi t1, t1, %pcrel_lo(.L4)
# RELOC: R_RISCV_PCREL_LO12_I .L4 0x0
# INSTR: addi t1, t1, %pcrel_lo(.L4)
# FIXUP: fixup A - offset: 0, value: %pcrel_lo(.L4), kind: fixup_riscv_pcrel_lo12_i

sb t1, %pcrel_lo(.L4)(a2)
# RELOC: R_RISCV_PCREL_LO12_S .L4 0x0
# INSTR: sb t1, %pcrel_lo(.L4)(a2)
# FIXUP: fixup A - offset: 0, value: %pcrel_lo(.L4), kind: fixup_riscv_pcrel_lo12_s

add t1, t1, tp, %tprel_add(foo)
# RELOC: R_RISCV_TPREL_ADD foo 0x0
# INSTR: add t1, t1, tp, %tprel_add(foo)
# FIXUP: fixup A - offset: 0, value: %tprel_add(foo), kind: fixup_riscv_tprel_add

jal zero, foo
# RELOC: R_RISCV_JAL
# INSTR: jal zero, foo
# FIXUP: fixup A - offset: 0, value: foo, kind: fixup_riscv_jal

bgeu a0, a1, foo
# RELOC: R_RISCV_BRANCH
# INSTR: bgeu a0, a1, foo
# FIXUP: fixup A - offset: 0, value: foo, kind: fixup_riscv_branch
