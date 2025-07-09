# RUN: llvm-mc -triple riscv32 -M no-aliases %s -show-encoding \
# RUN:     | FileCheck -check-prefix=INSTR %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+c %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=RELOC %s

# Check prefixes:
# RELOC - Check the relocation in the object.
# INSTR - Check the instruction is handled properly by the ASMPrinter

.long foo
# RELOC: R_RISCV_32 foo

.quad foo
# RELOC: R_RISCV_64 foo

lui t1, %hi(foo)
# RELOC: R_RISCV_HI20 foo 0x0
# INSTR: lui t1, %hi(foo)

lui t1, %hi(foo+4)
# RELOC: R_RISCV_HI20 foo 0x4
# INSTR: lui t1, %hi(foo+4)

lui t1, %tprel_hi(foo)
# RELOC: R_RISCV_TPREL_HI20 foo 0x0
# INSTR: lui t1, %tprel_hi(foo)

lui t1, %tprel_hi(foo+4)
# RELOC: R_RISCV_TPREL_HI20 foo 0x4
# INSTR: lui t1, %tprel_hi(foo+4)

addi t1, t1, %lo(foo)
# RELOC: R_RISCV_LO12_I foo 0x0
# INSTR: addi t1, t1, %lo(foo)

addi t1, t1, %lo(foo+4)
# RELOC: R_RISCV_LO12_I foo 0x4
# INSTR: addi t1, t1, %lo(foo+4)

addi t1, t1, %tprel_lo(foo)
# RELOC: R_RISCV_TPREL_LO12_I foo 0x0
# INSTR: addi t1, t1, %tprel_lo(foo)

addi t1, t1, %tprel_lo(foo+4)
# RELOC: R_RISCV_TPREL_LO12_I foo 0x4
# INSTR: addi t1, t1, %tprel_lo(foo+4)

sb t1, %lo(foo)(a2)
# RELOC: R_RISCV_LO12_S foo 0x0
# INSTR: sb t1, %lo(foo)(a2)

sb t1, %lo(foo+4)(a2)
# RELOC: R_RISCV_LO12_S foo 0x4
# INSTR: sb t1, %lo(foo+4)(a2)

sb t1, %tprel_lo(foo)(a2)
# RELOC: R_RISCV_TPREL_LO12_S foo 0x0
# INSTR: sb t1, %tprel_lo(foo)(a2)

sb t1, %tprel_lo(foo+4)(a2)
# RELOC: R_RISCV_TPREL_LO12_S foo 0x4
# INSTR: sb t1, %tprel_lo(foo+4)(a2)

.L0:
auipc t1, %pcrel_hi(foo)
# RELOC: R_RISCV_PCREL_HI20 foo 0x0
# INSTR: auipc t1, %pcrel_hi(foo)

auipc t1, %pcrel_hi(foo+4)
# RELOC: R_RISCV_PCREL_HI20 foo 0x4
# INSTR: auipc t1, %pcrel_hi(foo+4)

addi t1, t1, %pcrel_lo(.L0)
# RELOC: R_RISCV_PCREL_LO12_I .L0 0x0
# INSTR: addi t1, t1, %pcrel_lo(.L0)

sb t1, %pcrel_lo(.L0)(a2)
# RELOC: R_RISCV_PCREL_LO12_S .L0 0x0
# INSTR: sb t1, %pcrel_lo(.L0)(a2)

.L1:
auipc t1, %got_pcrel_hi(foo)
# RELOC: R_RISCV_GOT_HI20 foo 0x0
# INSTR: auipc t1, %got_pcrel_hi(foo)

addi t1, t1, %pcrel_lo(.L1)
# RELOC: R_RISCV_PCREL_LO12_I .L1 0x0
# INSTR: addi t1, t1, %pcrel_lo(.L1)

sb t1, %pcrel_lo(.L1)(a2)
# RELOC: R_RISCV_PCREL_LO12_S .L1 0x0
# INSTR: sb t1, %pcrel_lo(.L1)(a2)

# Check that GOT relocations aren't evaluated to a constant when the symbol is
# in the same object file.
.L2:
auipc t1, %got_pcrel_hi(.L1)
# RELOC: R_RISCV_GOT_HI20 .L1 0x0
# INSTR: auipc t1, %got_pcrel_hi(.L1)

addi t1, t1, %pcrel_lo(.L2)
# RELOC: R_RISCV_PCREL_LO12_I .L2 0x0
# INSTR: addi t1, t1, %pcrel_lo(.L2)

sb t1, %pcrel_lo(.L2)(a2)
# RELOC: R_RISCV_PCREL_LO12_S .L2 0x0
# INSTR: sb t1, %pcrel_lo(.L2)(a2)

.L3:
auipc t1, %tls_ie_pcrel_hi(foo)
# RELOC: R_RISCV_TLS_GOT_HI20 foo 0x0
# INSTR: auipc t1, %tls_ie_pcrel_hi(foo)

addi t1, t1, %pcrel_lo(.L3)
# RELOC: R_RISCV_PCREL_LO12_I .L3 0x0
# INSTR: addi t1, t1, %pcrel_lo(.L3)

sb t1, %pcrel_lo(.L3)(a2)
# RELOC: R_RISCV_PCREL_LO12_S .L3 0x0
# INSTR: sb t1, %pcrel_lo(.L3)(a2)

.L4:
auipc t1, %tls_gd_pcrel_hi(foo)
# RELOC: R_RISCV_TLS_GD_HI20 foo 0x0
# INSTR: auipc t1, %tls_gd_pcrel_hi(foo)

addi t1, t1, %pcrel_lo(.L4)
# RELOC: R_RISCV_PCREL_LO12_I .L4 0x0
# INSTR: addi t1, t1, %pcrel_lo(.L4)

sb t1, %pcrel_lo(.L4)(a2)
# RELOC: R_RISCV_PCREL_LO12_S .L4 0x0
# INSTR: sb t1, %pcrel_lo(.L4)(a2)

add t1, t1, tp, %tprel_add(foo)
# RELOC: R_RISCV_TPREL_ADD foo 0x0
# INSTR: add t1, t1, tp, %tprel_add(foo)

jal zero, foo
# RELOC: R_RISCV_JAL
# INSTR: jal zero, foo

# Since foo is undefined, this will be relaxed to (bltu; jal)
bgeu a0, a1, foo
# RELOC: R_RISCV_JAL
# INSTR: bgeu a0, a1, foo

.L5:
auipc a0, %tlsdesc_hi(a_symbol)
# RELOC: R_RISCV_TLSDESC_HI20
# INSTR: auipc a0, %tlsdesc_hi(a_symbol)

lw a1, %tlsdesc_load_lo(.L5)(a0)
# RELOC: R_RISCV_TLSDESC_LOAD_LO12
# INSTR: lw a1, %tlsdesc_load_lo(.L5)(a0)

addi a0, a0, %tlsdesc_add_lo(.L5)
# RELOC: R_RISCV_TLSDESC_ADD_LO12
# INSTR: addi a0, a0, %tlsdesc_add_lo(.L5)

jalr t0, 0(a1), %tlsdesc_call(.L5)
# RELOC: R_RISCV_TLSDESC_CALL
# INSTR: jalr t0, 0(a1), %tlsdesc_call(.L5)
