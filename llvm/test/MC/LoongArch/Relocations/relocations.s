# RUN: llvm-mc --triple=loongarch64 < %s --show-encoding \
# RUN:     | FileCheck --check-prefixes=INSTR,FIXUP %s
# RUN: llvm-mc --filetype=obj --triple=loongarch64 < %s \
# RUN:     | llvm-readobj -r - | FileCheck --check-prefix=RELOC %s

## Check prefixes:
## RELOC - Check the relocation in the object.
## FIXUP - Check the fixup on the instruction.
## INSTR - Check the instruction is handled properly by the ASMPrinter.

.long foo
# RELOC: R_LARCH_32 foo

.quad foo
# RELOC: R_LARCH_64 foo

pcalau12i $t1, %pc_hi20(foo)
# RELOC: R_LARCH_PCALA_HI20 foo 0x0
# INSTR: pcalau12i $t1, %pc_hi20(foo)
# FIXUP: fixup A - offset: 0, value: %pc_hi20(foo), kind: fixup_loongarch_pcala_hi20

pcalau12i $t1, %pc_hi20(foo+4)
# RELOC: R_LARCH_PCALA_HI20 foo 0x4
# INSTR: pcalau12i $t1, %pc_hi20(foo+4)
# FIXUP: fixup A - offset: 0, value: %pc_hi20(foo+4), kind: fixup_loongarch_pcala_hi20

addi.d $t1, $t1, %pc_lo12(foo)
# RELOC: R_LARCH_PCALA_LO12 foo 0x0
# INSTR: addi.d  $t1, $t1, %pc_lo12(foo)
# FIXUP: fixup A - offset: 0, value: %pc_lo12(foo), kind: fixup_loongarch_pcala_lo12

addi.d $t1, $t1, %pc_lo12(foo+4)
# RELOC: R_LARCH_PCALA_LO12 foo 0x4
# INSTR: addi.d  $t1, $t1, %pc_lo12(foo+4)
# FIXUP: fixup A - offset: 0, value: %pc_lo12(foo+4), kind: fixup_loongarch_pcala_lo12

st.b $t1, $a2, %pc_lo12(foo)
# RELOC: R_LARCH_PCALA_LO12 foo 0x0
# INSTR: st.b  $t1, $a2, %pc_lo12(foo)
# FIXUP: fixup A - offset: 0, value: %pc_lo12(foo), kind: fixup_loongarch_pcala_lo12

st.b $t1, $a2, %pc_lo12(foo+4)
# RELOC: R_LARCH_PCALA_LO12 foo 0x4
# INSTR: st.b  $t1, $a2, %pc_lo12(foo+4)
# FIXUP: fixup A - offset: 0, value: %pc_lo12(foo+4), kind: fixup_loongarch_pcala_lo12

bl %plt(foo)
# RELOC: R_LARCH_B26
# INSTR: bl  %plt(foo)
# FIXUP: fixup A - offset: 0, value: %plt(foo), kind: fixup_loongarch_b26

bl foo
# RELOC: R_LARCH_B26
# INSTR: bl  foo
# FIXUP: fixup A - offset: 0, value: foo, kind: fixup_loongarch_b26
