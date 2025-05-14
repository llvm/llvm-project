# RUN: llvm-mc --triple=loongarch64 %s --show-encoding \
# RUN:     | FileCheck --check-prefix=INSTR %s
# RUN: llvm-mc --filetype=obj --triple=loongarch64 %s \
# RUN:     | llvm-readobj -r - | FileCheck --check-prefix=RELOC %s

## Check prefixes:
## RELOC - Check the relocation in the object.
## INSTR - Check the instruction is handled properly by the ASMPrinter.

.long foo
# RELOC: R_LARCH_32 foo

.quad foo
# RELOC: R_LARCH_64 foo

bne $t1, $t2, %b16(foo)
# RELOC: R_LARCH_B16
# INSTR: bne $t1, $t2, %b16(foo)

bnez $t1, %b21(foo)
# RELOC: R_LARCH_B21
# INSTR: bnez $t1, %b21(foo)

bl %plt(foo)
# RELOC: R_LARCH_B26
# INSTR: bl foo

bl foo
# RELOC: R_LARCH_B26
# INSTR: bl foo

lu12i.w $t1, %abs_hi20(foo)
# RELOC: R_LARCH_ABS_HI20 foo 0x0
# INSTR: lu12i.w $t1, %abs_hi20(foo)

ori $t1, $t1, %abs_lo12(foo)
# RELOC: R_LARCH_ABS_LO12 foo 0x0
# INSTR: ori $t1, $t1, %abs_lo12(foo)

lu32i.d $t1, %abs64_lo20(foo)
# RELOC: R_LARCH_ABS64_LO20 foo 0x0
# INSTR: lu32i.d $t1, %abs64_lo20(foo)

lu52i.d $t1, $t1, %abs64_hi12(foo)
# RELOC: R_LARCH_ABS64_HI12 foo 0x0
# INSTR: lu52i.d $t1, $t1, %abs64_hi12(foo)

pcalau12i $t1, %pc_hi20(foo)
# RELOC: R_LARCH_PCALA_HI20 foo 0x0
# INSTR: pcalau12i $t1, %pc_hi20(foo)

pcalau12i $t1, %pc_hi20(foo+4)
# RELOC: R_LARCH_PCALA_HI20 foo 0x4
# INSTR: pcalau12i $t1, %pc_hi20(foo+4)

addi.d $t1, $t1, %pc_lo12(foo)
# RELOC: R_LARCH_PCALA_LO12 foo 0x0
# INSTR: addi.d  $t1, $t1, %pc_lo12(foo)

addi.d $t1, $t1, %pc_lo12(foo+4)
# RELOC: R_LARCH_PCALA_LO12 foo 0x4
# INSTR: addi.d  $t1, $t1, %pc_lo12(foo+4)

jirl $zero, $t1, %pc_lo12(foo)
# RELOC: R_LARCH_PCALA_LO12 foo 0x0
# INSTR: jirl $zero, $t1, %pc_lo12(foo)

st.b $t1, $a2, %pc_lo12(foo)
# RELOC: R_LARCH_PCALA_LO12 foo 0x0
# INSTR: st.b  $t1, $a2, %pc_lo12(foo)

st.b $t1, $a2, %pc_lo12(foo+4)
# RELOC: R_LARCH_PCALA_LO12 foo 0x4
# INSTR: st.b  $t1, $a2, %pc_lo12(foo+4)

lu32i.d $t1, %pc64_lo20(foo)
# RELOC: R_LARCH_PCALA64_LO20 foo 0x0
# INSTR: lu32i.d $t1, %pc64_lo20(foo)

lu52i.d $t1, $t1, %pc64_hi12(foo)
# RELOC: R_LARCH_PCALA64_HI12 foo 0x0
# INSTR: lu52i.d $t1, $t1, %pc64_hi12(foo)

pcalau12i $t1, %got_pc_hi20(foo)
# RELOC: R_LARCH_GOT_PC_HI20 foo 0x0
# INSTR: pcalau12i $t1, %got_pc_hi20(foo)

ld.d $t1, $a2, %got_pc_lo12(foo)
# RELOC: R_LARCH_GOT_PC_LO12 foo 0x0
# INSTR: ld.d  $t1, $a2, %got_pc_lo12(foo)

lu32i.d $t1, %got64_pc_lo20(foo)
# RELOC: R_LARCH_GOT64_PC_LO20 foo 0x0
# INSTR: lu32i.d $t1, %got64_pc_lo20(foo)

lu52i.d $t1, $t1, %got64_pc_hi12(foo)
# RELOC: R_LARCH_GOT64_PC_HI12 foo 0x0
# INSTR: lu52i.d $t1, $t1, %got64_pc_hi12(foo)

lu12i.w $t1, %got_hi20(foo)
# RELOC: R_LARCH_GOT_HI20 foo 0x0
# INSTR: lu12i.w $t1, %got_hi20(foo)

ori $t1, $a2, %got_lo12(foo)
# RELOC: R_LARCH_GOT_LO12 foo 0x0
# INSTR: ori  $t1, $a2, %got_lo12(foo)

lu32i.d $t1, %got64_lo20(foo)
# RELOC: R_LARCH_GOT64_LO20 foo 0x0
# INSTR: lu32i.d $t1, %got64_lo20(foo)

lu52i.d $t1, $t1, %got64_hi12(foo)
# RELOC: R_LARCH_GOT64_HI12 foo 0x0
# INSTR: lu52i.d $t1, $t1, %got64_hi12(foo)

lu12i.w $t1, %le_hi20(foo)
# RELOC: R_LARCH_TLS_LE_HI20 foo 0x0
# INSTR: lu12i.w $t1, %le_hi20(foo)

ori $t1, $a2, %le_lo12(foo)
# RELOC: R_LARCH_TLS_LE_LO12 foo 0x0
# INSTR: ori  $t1, $a2, %le_lo12(foo)

lu32i.d $t1, %le64_lo20(foo)
# RELOC: R_LARCH_TLS_LE64_LO20 foo 0x0
# INSTR: lu32i.d $t1, %le64_lo20(foo)

lu52i.d $t1, $t1, %le64_hi12(foo)
# RELOC: R_LARCH_TLS_LE64_HI12 foo 0x0
# INSTR: lu52i.d $t1, $t1, %le64_hi12(foo)

pcalau12i $t1, %ie_pc_hi20(foo)
# RELOC: R_LARCH_TLS_IE_PC_HI20 foo 0x0
# INSTR: pcalau12i $t1, %ie_pc_hi20(foo)

ld.d $t1, $a2, %ie_pc_lo12(foo)
# RELOC: R_LARCH_TLS_IE_PC_LO12 foo 0x0
# INSTR: ld.d  $t1, $a2, %ie_pc_lo12(foo)

lu32i.d $t1, %ie64_pc_lo20(foo)
# RELOC: R_LARCH_TLS_IE64_PC_LO20 foo 0x0
# INSTR: lu32i.d $t1, %ie64_pc_lo20(foo)

lu52i.d $t1, $t1, %ie64_pc_hi12(foo)
# RELOC: R_LARCH_TLS_IE64_PC_HI12 foo 0x0
# INSTR: lu52i.d $t1, $t1, %ie64_pc_hi12(foo)

lu12i.w $t1, %ie_hi20(foo)
# RELOC: R_LARCH_TLS_IE_HI20 foo 0x0
# INSTR: lu12i.w $t1, %ie_hi20(foo)

ori $t1, $a2, %ie_lo12(foo)
# RELOC: R_LARCH_TLS_IE_LO12 foo 0x0
# INSTR: ori  $t1, $a2, %ie_lo12(foo)

lu32i.d $t1, %ie64_lo20(foo)
# RELOC: R_LARCH_TLS_IE64_LO20 foo 0x0
# INSTR: lu32i.d $t1, %ie64_lo20(foo)

lu52i.d $t1, $t1, %ie64_hi12(foo)
# RELOC: R_LARCH_TLS_IE64_HI12 foo 0x0
# INSTR: lu52i.d $t1, $t1, %ie64_hi12(foo)

pcalau12i $t1, %ld_pc_hi20(foo)
# RELOC: R_LARCH_TLS_LD_PC_HI20 foo 0x0
# INSTR: pcalau12i $t1, %ld_pc_hi20(foo)

lu12i.w $t1, %ld_hi20(foo)
# RELOC: R_LARCH_TLS_LD_HI20 foo 0x0
# INSTR: lu12i.w $t1, %ld_hi20(foo)

pcalau12i $t1, %gd_pc_hi20(foo)
# RELOC: R_LARCH_TLS_GD_PC_HI20 foo 0x0
# INSTR: pcalau12i $t1, %gd_pc_hi20(foo)

lu12i.w $t1, %gd_hi20(foo)
# RELOC: R_LARCH_TLS_GD_HI20 foo 0x0
# INSTR: lu12i.w $t1, %gd_hi20(foo)

pcaddu18i $t1, %call36(foo)
# RELOC: R_LARCH_CALL36 foo 0x0
# INSTR: pcaddu18i $t1, %call36(foo)

pcalau12i $t1, %desc_pc_hi20(foo)
# RELOC: R_LARCH_TLS_DESC_PC_HI20 foo 0x0
# INSTR: pcalau12i $t1, %desc_pc_hi20(foo)

addi.d $t1, $t1, %desc_pc_lo12(foo)
# RELOC: R_LARCH_TLS_DESC_PC_LO12 foo 0x0
# INSTR: addi.d $t1, $t1, %desc_pc_lo12(foo)

lu32i.d $t1, %desc64_pc_lo20(foo)
# RELOC: R_LARCH_TLS_DESC64_PC_LO20 foo 0x0
# INSTR: lu32i.d $t1, %desc64_pc_lo20(foo)

lu52i.d $t1, $t1, %desc64_pc_hi12(foo)
# RELOC: R_LARCH_TLS_DESC64_PC_HI12 foo 0x0
# INSTR: lu52i.d $t1, $t1, %desc64_pc_hi12(foo)

ld.d $ra, $t1, %desc_ld(foo)
# RELOC: R_LARCH_TLS_DESC_LD foo 0x0
# INSTR: ld.d $ra, $t1, %desc_ld(foo)

jirl $ra, $ra, %desc_call(foo)
# RELOC: R_LARCH_TLS_DESC_CALL foo 0x0
# INSTR: jirl $ra, $ra, %desc_call(foo)

lu12i.w $t1, %desc_hi20(foo)
# RELOC: R_LARCH_TLS_DESC_HI20 foo 0x0
# INSTR: lu12i.w $t1, %desc_hi20(foo)

ori $t1, $t1, %desc_lo12(foo)
# RELOC: R_LARCH_TLS_DESC_LO12 foo 0x0
# INSTR: ori $t1, $t1, %desc_lo12(foo)

lu32i.d $t1, %desc64_lo20(foo)
# RELOC: R_LARCH_TLS_DESC64_LO20 foo 0x0
# INSTR: lu32i.d $t1, %desc64_lo20(foo)

lu52i.d $t1, $t1, %desc64_hi12(foo)
# RELOC: R_LARCH_TLS_DESC64_HI12 foo 0x0
# INSTR: lu52i.d $t1, $t1, %desc64_hi12(foo)

lu12i.w $t1, %le_hi20_r(foo)
# RELOC: R_LARCH_TLS_LE_HI20_R foo 0x0
# INSTR: lu12i.w $t1, %le_hi20_r(foo)

add.d $t1, $t2, $tp, %le_add_r(foo)
# RELOC: R_LARCH_TLS_LE_ADD_R foo 0x0
# INSTR: add.d $t1, $t2, $tp, %le_add_r(foo)

addi.d $t1, $a2, %le_lo12_r(foo)
# RELOC: R_LARCH_TLS_LE_LO12_R foo 0x0
# INSTR: addi.d $t1, $a2, %le_lo12_r(foo)

pcaddi $t1, %pcrel_20(foo)
# RELOC: R_LARCH_PCREL20_S2 foo 0x0
# INSTR: pcaddi $t1, %pcrel_20(foo)

pcaddi $t1, %ld_pcrel_20(foo)
# RELOC: R_LARCH_TLS_LD_PCREL20_S2 foo 0x0
# INSTR: pcaddi $t1, %ld_pcrel_20(foo)

pcaddi $t1, %gd_pcrel_20(foo)
# RELOC: R_LARCH_TLS_GD_PCREL20_S2 foo 0x0
# INSTR: pcaddi $t1, %gd_pcrel_20(foo)

pcaddi $t1, %desc_pcrel_20(foo)
# RELOC: R_LARCH_TLS_DESC_PCREL20_S2 foo 0x0
# INSTR: pcaddi $t1, %desc_pcrel_20(foo)

fld.s $ft1, $a0, %pc_lo12(foo)
# RELOC: R_LARCH_PCALA_LO12 foo 0x0
# INSTR: fld.s $ft1, $a0, %pc_lo12(foo)

fst.d $ft1, $a0, %pc_lo12(foo)
# RELOC: R_LARCH_PCALA_LO12 foo 0x0
# INSTR: fst.d $ft1, $a0, %pc_lo12(foo)

vld $vr9, $a0, %pc_lo12(foo)
# RELOC: R_LARCH_PCALA_LO12 foo 0x0
# INSTR: vld $vr9, $a0, %pc_lo12(foo)

vst $vr9, $a0, %pc_lo12(foo)
# RELOC: R_LARCH_PCALA_LO12 foo 0x0
# INSTR: vst $vr9, $a0, %pc_lo12(foo)

xvld $xr9, $a0, %pc_lo12(foo)
# RELOC: R_LARCH_PCALA_LO12 foo 0x0
# INSTR: xvld $xr9, $a0, %pc_lo12(foo)

xvst $xr9, $a0, %pc_lo12(foo)
# RELOC: R_LARCH_PCALA_LO12 foo 0x0
# INSTR: xvst $xr9, $a0, %pc_lo12(foo)
