## Test la/la.global/la.local expand to different instructions sequence under
## different features.

# RUN: llvm-mc --triple=loongarch64 %s \
# RUN:     | FileCheck %s --check-prefix=NORMAL
# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=-relax %s -o %t
# RUN: llvm-readobj -r %t | FileCheck %s --check-prefix=RELOC
# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=+relax %s -o %t.relax
# RUN: llvm-readobj -r %t.relax | FileCheck %s --check-prefixes=RELOC,RELAX
# RUN: llvm-mc --triple=loongarch64 --mattr=+la-global-with-pcrel < %s \
# RUN:     | FileCheck %s --check-prefix=GTOPCR
# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=+la-global-with-pcrel \
# RUN:     --mattr=-relax %s -o %t
# RUN: llvm-readobj -r %t | FileCheck %s --check-prefix=GTOPCR-RELOC
# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=+la-global-with-pcrel \
# RUN:     --mattr=+relax %s -o %t.relax
# RUN: llvm-readobj -r %t.relax | FileCheck %s --check-prefixes=GTOPCR-RELOC,GTOPCR-RELAX
# RUN: llvm-mc --triple=loongarch64 --mattr=+la-global-with-abs < %s \
# RUN:     | FileCheck %s --check-prefix=GTOABS
# RUN: llvm-mc --triple=loongarch64 --mattr=+la-local-with-abs < %s \
# RUN:     | FileCheck %s --check-prefix=LTOABS

# RELOC:      Relocations [
# RELOC-NEXT:   Section ({{.*}}) .rela.text {

la $a0, sym
# NORMAL:      pcalau12i $a0, %got_pc_hi20(sym)
# NORMAL-NEXT: ld.d $a0, $a0, %got_pc_lo12(sym)

# GTOPCR:      pcalau12i $a0, %pc_hi20(sym)
# GTOPCR-NEXT: addi.d $a0, $a0, %pc_lo12(sym)

# GTOABS:      lu12i.w $a0, %abs_hi20(sym)
# GTOABS-NEXT: ori $a0, $a0, %abs_lo12(sym)
# GTOABS-NEXT: lu32i.d $a0, %abs64_lo20(sym)
# GTOABS-NEXT: lu52i.d $a0, $a0, %abs64_hi12(sym)

# RELOC-NEXT: R_LARCH_GOT_PC_HI20 sym 0x0
# RELAX-NEXT: R_LARCH_RELAX - 0x0
# RELOC-NEXT: R_LARCH_GOT_PC_LO12 sym 0x0
# RELAX-NEXT: R_LARCH_RELAX - 0x0

# GTOPCR-RELOC: R_LARCH_PCALA_HI20 sym 0x0
# GTOPCR-RELAX: R_LARCH_RELAX - 0x0
# GTOPCR-RELOC-NEXT: R_LARCH_PCALA_LO12 sym 0x0
# GTOPCR-RELAX-NEXT: R_LARCH_RELAX - 0x0

la.global $a0, sym_global
# NORMAL:      pcalau12i $a0, %got_pc_hi20(sym_global)
# NORMAL-NEXT: ld.d $a0, $a0, %got_pc_lo12(sym_global)

# GTOPCR:      pcalau12i $a0, %pc_hi20(sym_global)
# GTOPCR-NEXT: addi.d $a0, $a0, %pc_lo12(sym_global)

# GTOABS:      lu12i.w $a0, %abs_hi20(sym_global)
# GTOABS-NEXT: ori $a0, $a0, %abs_lo12(sym_global)
# GTOABS-NEXT: lu32i.d $a0, %abs64_lo20(sym_global)
# GTOABS-NEXT: lu52i.d $a0, $a0, %abs64_hi12(sym_global)

# RELOC-NEXT: R_LARCH_GOT_PC_HI20 sym_global 0x0
# RELAX-NEXT: R_LARCH_RELAX - 0x0
# RELOC-NEXT: R_LARCH_GOT_PC_LO12 sym_global 0x0
# RELAX-NEXT: R_LARCH_RELAX - 0x0

# GTOPCR-RELOC-NEXT: R_LARCH_PCALA_HI20 sym_global 0x0
# GTOPCR-RELAX-NEXT: R_LARCH_RELAX - 0x0
# GTOPCR-RELOC-NEXT: R_LARCH_PCALA_LO12 sym_global 0x0
# GTOPCR-RELAX-NEXT: R_LARCH_RELAX - 0x0

la.global $a0, $a1, sym_global_large
# NORMAL:      pcalau12i $a0, %got_pc_hi20(sym_global_large)
# NORMAL-NEXT: addi.d $a1, $zero, %got_pc_lo12(sym_global_large)
# NORMAL-NEXT: lu32i.d $a1, %got64_pc_lo20(sym_global_large)
# NORMAL-NEXT: lu52i.d $a1, $a1, %got64_pc_hi12(sym_global_large)
# NORMAL-NEXT: ldx.d $a0, $a0, $a1

# GTOPCR:      pcalau12i $a0, %pc_hi20(sym_global_large)
# GTOPCR-NEXT: addi.d $a1, $zero, %pc_lo12(sym_global_large)
# GTOPCR-NEXT: lu32i.d $a1, %pc64_lo20(sym_global_large)
# GTOPCR-NEXT: lu52i.d $a1, $a1, %pc64_hi12(sym_global_large)
# GTOPCR-NEXT: add.d $a0, $a0, $a1

# GTOABS:      lu12i.w $a0, %abs_hi20(sym_global_large)
# GTOABS-NEXT: ori $a0, $a0, %abs_lo12(sym_global_large)
# GTOABS-NEXT: lu32i.d $a0, %abs64_lo20(sym_global_large)
# GTOABS-NEXT: lu52i.d $a0, $a0, %abs64_hi12(sym_global_large)

# RELOC-NEXT: R_LARCH_GOT_PC_HI20 sym_global_large 0x0
# RELOC-NEXT: R_LARCH_GOT_PC_LO12 sym_global_large 0x0
# RELOC-NEXT: R_LARCH_GOT64_PC_LO20 sym_global_large 0x0
# RELOC-NEXT: R_LARCH_GOT64_PC_HI12 sym_global_large 0x0

la.local $a0, sym_local
# NORMAL:      pcalau12i $a0, %pc_hi20(sym_local)
# NORMAL-NEXT: addi.d $a0, $a0, %pc_lo12(sym_local)

# LTOABS:      lu12i.w $a0, %abs_hi20(sym_local)
# LTOABS-NEXT: ori $a0, $a0, %abs_lo12(sym_local)
# LTOABS-NEXT: lu32i.d $a0, %abs64_lo20(sym_local)
# LTOABS-NEXT: lu52i.d $a0, $a0, %abs64_hi12(sym_local)

# RELOC-NEXT: R_LARCH_PCALA_HI20 sym_local 0x0
# RELAX-NEXT: R_LARCH_RELAX - 0x0
# RELOC-NEXT: R_LARCH_PCALA_LO12 sym_local 0x0
# RELAX-NEXT: R_LARCH_RELAX - 0x0

la.local $a0, $a1, sym_local_large
# NORMAL:      pcalau12i $a0, %pc_hi20(sym_local_large)
# NORMAL-NEXT: addi.d $a1, $zero, %pc_lo12(sym_local_large)
# NORMAL-NEXT: lu32i.d $a1, %pc64_lo20(sym_local_large)
# NORMAL-NEXT: lu52i.d $a1, $a1, %pc64_hi12(sym_local_large)
# NORMAL-NEXT: add.d $a0, $a0, $a1

# LTOABS:      lu12i.w $a0, %abs_hi20(sym_local_large)
# LTOABS-NEXT: ori $a0, $a0, %abs_lo12(sym_local_large)
# LTOABS-NEXT: lu32i.d $a0, %abs64_lo20(sym_local_large)
# LTOABS-NEXT: lu52i.d $a0, $a0, %abs64_hi12(sym_local_large)

# RELOC-NEXT: R_LARCH_PCALA_HI20 sym_local_large 0x0
# RELOC-NEXT: R_LARCH_PCALA_LO12 sym_local_large 0x0
# RELOC-NEXT: R_LARCH_PCALA64_LO20 sym_local_large 0x0
# RELOC-NEXT: R_LARCH_PCALA64_HI12 sym_local_large 0x0


# RELOC-NEXT:   }
# RELOC-NEXT: ]
