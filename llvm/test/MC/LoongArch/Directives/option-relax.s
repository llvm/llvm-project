# RUN: llvm-mc --triple=loongarch64 %s | FileCheck --check-prefix=CHECK-ASM %s
# RUN: llvm-mc -filetype=obj --triple=loongarch64 %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=CHECK-RELOC %s

## Check .option relax causes R_LARCH_RELAX to be emitted, and .option
## norelax suppresses it.

# CHECK-ASM: .option relax
.option relax

# CHECK-ASM: pcalau12i $a0, %pc_hi20(sym1)
# CHECK-ASM-NEXT: addi.d $a0, $a0, %pc_lo12(sym1)

# CHECK-RELOC:      R_LARCH_PCALA_HI20 sym1 0x0
# CHECK-RELOC-NEXT: R_LARCH_RELAX - 0x0
# CHECK-RELOC-NEXT: R_LARCH_PCALA_LO12 sym1 0x0
# CHECK-RELOC-NEXT: R_LARCH_RELAX - 0x0
la.pcrel $a0, sym1

# CHECK-ASM: .option norelax
.option norelax

# CHECK-ASM: pcalau12i $a0, %pc_hi20(sym2)
# CHECK-ASM-NEXT: addi.d $a0, $a0, %pc_lo12(sym2)

# CHECK-RELOC-NEXT: R_LARCH_PCALA_HI20 sym2 0x0
# CHECK-RELOC-NOT:  R_LARCH_RELAX - 0x0
# CHECK-RELOC-NEXT: R_LARCH_PCALA_LO12 sym2 0x0
# CHECK-RELOC-NOT:  R_LARCH_RELAX - 0x0
la.pcrel $a0, sym2
