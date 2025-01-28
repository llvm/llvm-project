# RUN: llvm-mc --triple=loongarch64 --mattr=-relax %s \
# RUN:     | FileCheck --check-prefix=CHECK-ASM %s
# RUN: llvm-mc --triple=loongarch64 --mattr=-relax --filetype=obj %s \
# RUN:     | llvm-readobj -r - | FileCheck --check-prefix=CHECK-RELOC %s

## Test the operation of the push and pop assembler directives when
## using .option relax. Checks that using .option pop correctly restores 
## all target features to their state at the point where .option pop was
## last used.

# CHECK-ASM: .option push
.option push # relax = false

# CHECK-ASM: .option relax
.option relax # relax = true

# CHECK-ASM: pcalau12i $a0, %pc_hi20(sym1)
# CHECK-ASM-NEXT: addi.d $a0, $a0, %pc_lo12(sym1)
# CHECK-RELOC:      R_LARCH_PCALA_HI20 sym1 0x0
# CHECK-RELOC-NEXT: R_LARCH_RELAX - 0x0
# CHECK-RELOC-NEXT: R_LARCH_PCALA_LO12 sym1 0x0
# CHECK-RELOC-NEXT: R_LARCH_RELAX - 0x0
la.pcrel $a0, sym1

# CHECK-ASM: .option push
.option push # relax = true

# CHECK-ASM: .option norelax
.option norelax # relax = false

# CHECK-ASM: pcalau12i $a0, %pc_hi20(sym2)
# CHECK-ASM-NEXT: addi.d $a0, $a0, %pc_lo12(sym2)
# CHECK-RELOC-NEXT: R_LARCH_PCALA_HI20 sym2 0x0
# CHECK-RELOC-NOT:  R_LARCH_RELAX - 0x0
# CHECK-RELOC-NEXT: R_LARCH_PCALA_LO12 sym2 0x0
# CHECK-RELOC-NOT:  R_LARCH_RELAX - 0x0
la.pcrel $a0, sym2

# CHECK-ASM: .option pop
.option pop # relax = true

# CHECK-ASM: pcalau12i $a0, %pc_hi20(sym3)
# CHECK-ASM-NEXT: addi.d $a0, $a0, %pc_lo12(sym3)
# CHECK-RELOC:      R_LARCH_PCALA_HI20 sym3 0x0
# CHECK-RELOC-NEXT: R_LARCH_RELAX - 0x0
# CHECK-RELOC-NEXT: R_LARCH_PCALA_LO12 sym3 0x0
# CHECK-RELOC-NEXT: R_LARCH_RELAX - 0x0
la.pcrel $a0, sym3

# CHECK-ASM: .option pop
.option pop # relax = false

la.pcrel $a0, sym4
# CHECK-ASM: pcalau12i $a0, %pc_hi20(sym4)
# CHECK-ASM-NEXT: addi.d $a0, $a0, %pc_lo12(sym4)
# CHECK-RELOC-NEXT: R_LARCH_PCALA_HI20 sym4 0x0
# CHECK-RELOC-NOT:  R_LARCH_RELAX - 0x0
# CHECK-RELOC-NEXT: R_LARCH_PCALA_LO12 sym4 0x0
# CHECK-RELOC-NOT:  R_LARCH_RELAX - 0x0
