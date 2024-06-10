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

bne $t1, $t2, %b16(foo)
# RELOC: R_LARCH_B16
# INSTR: bne $t1, $t2, %b16(foo)
# FIXUP: fixup A - offset: 0, value: %b16(foo), kind: fixup_loongarch_b16

bnez $t1, %b21(foo)
# RELOC: R_LARCH_B21
# INSTR: bnez $t1, %b21(foo)
# FIXUP: fixup A - offset: 0, value: %b21(foo), kind: fixup_loongarch_b21

bl %plt(foo)
# RELOC: R_LARCH_B26
# INSTR: bl %plt(foo)
# FIXUP: fixup A - offset: 0, value: %plt(foo), kind: fixup_loongarch_b26

bl foo
# RELOC: R_LARCH_B26
# INSTR: bl foo
# FIXUP: fixup A - offset: 0, value: foo, kind: fixup_loongarch_b26

lu12i.w $t1, %abs_hi20(foo)
# RELOC: R_LARCH_ABS_HI20 foo 0x0
# INSTR: lu12i.w $t1, %abs_hi20(foo)
# FIXUP: fixup A - offset: 0, value: %abs_hi20(foo), kind: fixup_loongarch_abs_hi20

ori $t1, $t1, %abs_lo12(foo)
# RELOC: R_LARCH_ABS_LO12 foo 0x0
# INSTR: ori $t1, $t1, %abs_lo12(foo)
# FIXUP: fixup A - offset: 0, value: %abs_lo12(foo), kind: fixup_loongarch_abs_lo12

lu32i.d $t1, %abs64_lo20(foo)
# RELOC: R_LARCH_ABS64_LO20 foo 0x0
# INSTR: lu32i.d $t1, %abs64_lo20(foo)
# FIXUP: fixup A - offset: 0, value: %abs64_lo20(foo), kind: fixup_loongarch_abs64_lo20

lu52i.d $t1, $t1, %abs64_hi12(foo)
# RELOC: R_LARCH_ABS64_HI12 foo 0x0
# INSTR: lu52i.d $t1, $t1, %abs64_hi12(foo)
# FIXUP: fixup A - offset: 0, value: %abs64_hi12(foo), kind: fixup_loongarch_abs64_hi12

pcalau12i $t1, %pc_hi20(foo)
# RELOC: R_LARCH_PCALA_HI20 foo 0x0
# INSTR: pcalau12i $t1, %pc_hi20(foo)
# FIXUP: fixup A - offset: 0, value: %pc_hi20(foo), kind: FK_NONE

pcalau12i $t1, %pc_hi20(foo+4)
# RELOC: R_LARCH_PCALA_HI20 foo 0x4
# INSTR: pcalau12i $t1, %pc_hi20(foo+4)
# FIXUP: fixup A - offset: 0, value: %pc_hi20(foo+4), kind: FK_NONE

addi.d $t1, $t1, %pc_lo12(foo)
# RELOC: R_LARCH_PCALA_LO12 foo 0x0
# INSTR: addi.d  $t1, $t1, %pc_lo12(foo)
# FIXUP: fixup A - offset: 0, value: %pc_lo12(foo), kind: FK_NONE

addi.d $t1, $t1, %pc_lo12(foo+4)
# RELOC: R_LARCH_PCALA_LO12 foo 0x4
# INSTR: addi.d  $t1, $t1, %pc_lo12(foo+4)
# FIXUP: fixup A - offset: 0, value: %pc_lo12(foo+4), kind: FK_NONE

jirl $zero, $t1, %pc_lo12(foo)
# RELOC: R_LARCH_PCALA_LO12 foo 0x0
# INSTR: jirl $zero, $t1, %pc_lo12(foo)
# FIXUP: fixup A - offset: 0, value: %pc_lo12(foo), kind: FK_NONE

st.b $t1, $a2, %pc_lo12(foo)
# RELOC: R_LARCH_PCALA_LO12 foo 0x0
# INSTR: st.b  $t1, $a2, %pc_lo12(foo)
# FIXUP: fixup A - offset: 0, value: %pc_lo12(foo), kind: FK_NONE

st.b $t1, $a2, %pc_lo12(foo+4)
# RELOC: R_LARCH_PCALA_LO12 foo 0x4
# INSTR: st.b  $t1, $a2, %pc_lo12(foo+4)
# FIXUP: fixup A - offset: 0, value: %pc_lo12(foo+4), kind: FK_NONE

lu32i.d $t1, %pc64_lo20(foo)
# RELOC: R_LARCH_PCALA64_LO20 foo 0x0
# INSTR: lu32i.d $t1, %pc64_lo20(foo)
# FIXUP: fixup A - offset: 0, value: %pc64_lo20(foo), kind: FK_NONE

lu52i.d $t1, $t1, %pc64_hi12(foo)
# RELOC: R_LARCH_PCALA64_HI12 foo 0x0
# INSTR: lu52i.d $t1, $t1, %pc64_hi12(foo)
# FIXUP: fixup A - offset: 0, value: %pc64_hi12(foo), kind: FK_NONE

pcalau12i $t1, %got_pc_hi20(foo)
# RELOC: R_LARCH_GOT_PC_HI20 foo 0x0
# INSTR: pcalau12i $t1, %got_pc_hi20(foo)
# FIXUP: fixup A - offset: 0, value: %got_pc_hi20(foo), kind: FK_NONE

ld.d $t1, $a2, %got_pc_lo12(foo)
# RELOC: R_LARCH_GOT_PC_LO12 foo 0x0
# INSTR: ld.d  $t1, $a2, %got_pc_lo12(foo)
# FIXUP: fixup A - offset: 0, value: %got_pc_lo12(foo), kind: FK_NONE

lu32i.d $t1, %got64_pc_lo20(foo)
# RELOC: R_LARCH_GOT64_PC_LO20 foo 0x0
# INSTR: lu32i.d $t1, %got64_pc_lo20(foo)
# FIXUP: fixup A - offset: 0, value: %got64_pc_lo20(foo), kind: FK_NONE

lu52i.d $t1, $t1, %got64_pc_hi12(foo)
# RELOC: R_LARCH_GOT64_PC_HI12 foo 0x0
# INSTR: lu52i.d $t1, $t1, %got64_pc_hi12(foo)
# FIXUP: fixup A - offset: 0, value: %got64_pc_hi12(foo), kind: FK_NONE

lu12i.w $t1, %got_hi20(foo)
# RELOC: R_LARCH_GOT_HI20 foo 0x0
# INSTR: lu12i.w $t1, %got_hi20(foo)
# FIXUP: fixup A - offset: 0, value: %got_hi20(foo), kind: FK_NONE

ori $t1, $a2, %got_lo12(foo)
# RELOC: R_LARCH_GOT_LO12 foo 0x0
# INSTR: ori  $t1, $a2, %got_lo12(foo)
# FIXUP: fixup A - offset: 0, value: %got_lo12(foo), kind: FK_NONE

lu32i.d $t1, %got64_lo20(foo)
# RELOC: R_LARCH_GOT64_LO20 foo 0x0
# INSTR: lu32i.d $t1, %got64_lo20(foo)
# FIXUP: fixup A - offset: 0, value: %got64_lo20(foo), kind: FK_NONE

lu52i.d $t1, $t1, %got64_hi12(foo)
# RELOC: R_LARCH_GOT64_HI12 foo 0x0
# INSTR: lu52i.d $t1, $t1, %got64_hi12(foo)
# FIXUP: fixup A - offset: 0, value: %got64_hi12(foo), kind: FK_NONE

lu12i.w $t1, %le_hi20(foo)
# RELOC: R_LARCH_TLS_LE_HI20 foo 0x0
# INSTR: lu12i.w $t1, %le_hi20(foo)
# FIXUP: fixup A - offset: 0, value: %le_hi20(foo), kind: fixup_loongarch_tls_le_hi20

ori $t1, $a2, %le_lo12(foo)
# RELOC: R_LARCH_TLS_LE_LO12 foo 0x0
# INSTR: ori  $t1, $a2, %le_lo12(foo)
# FIXUP: fixup A - offset: 0, value: %le_lo12(foo), kind: fixup_loongarch_tls_le_lo12

lu32i.d $t1, %le64_lo20(foo)
# RELOC: R_LARCH_TLS_LE64_LO20 foo 0x0
# INSTR: lu32i.d $t1, %le64_lo20(foo)
# FIXUP: fixup A - offset: 0, value: %le64_lo20(foo), kind: fixup_loongarch_tls_le64_lo20

lu52i.d $t1, $t1, %le64_hi12(foo)
# RELOC: R_LARCH_TLS_LE64_HI12 foo 0x0
# INSTR: lu52i.d $t1, $t1, %le64_hi12(foo)
# FIXUP: fixup A - offset: 0, value: %le64_hi12(foo), kind: fixup_loongarch_tls_le64_hi12

pcalau12i $t1, %ie_pc_hi20(foo)
# RELOC: R_LARCH_TLS_IE_PC_HI20 foo 0x0
# INSTR: pcalau12i $t1, %ie_pc_hi20(foo)
# FIXUP: fixup A - offset: 0, value: %ie_pc_hi20(foo), kind: FK_NONE

ld.d $t1, $a2, %ie_pc_lo12(foo)
# RELOC: R_LARCH_TLS_IE_PC_LO12 foo 0x0
# INSTR: ld.d  $t1, $a2, %ie_pc_lo12(foo)
# FIXUP: fixup A - offset: 0, value: %ie_pc_lo12(foo), kind: FK_NONE

lu32i.d $t1, %ie64_pc_lo20(foo)
# RELOC: R_LARCH_TLS_IE64_PC_LO20 foo 0x0
# INSTR: lu32i.d $t1, %ie64_pc_lo20(foo)
# FIXUP: fixup A - offset: 0, value: %ie64_pc_lo20(foo), kind: FK_NONE

lu52i.d $t1, $t1, %ie64_pc_hi12(foo)
# RELOC: R_LARCH_TLS_IE64_PC_HI12 foo 0x0
# INSTR: lu52i.d $t1, $t1, %ie64_pc_hi12(foo)
# FIXUP: fixup A - offset: 0, value: %ie64_pc_hi12(foo), kind: FK_NONE

lu12i.w $t1, %ie_hi20(foo)
# RELOC: R_LARCH_TLS_IE_HI20 foo 0x0
# INSTR: lu12i.w $t1, %ie_hi20(foo)
# FIXUP: fixup A - offset: 0, value: %ie_hi20(foo), kind: FK_NONE

ori $t1, $a2, %ie_lo12(foo)
# RELOC: R_LARCH_TLS_IE_LO12 foo 0x0
# INSTR: ori  $t1, $a2, %ie_lo12(foo)
# FIXUP: fixup A - offset: 0, value: %ie_lo12(foo), kind: FK_NONE

lu32i.d $t1, %ie64_lo20(foo)
# RELOC: R_LARCH_TLS_IE64_LO20 foo 0x0
# INSTR: lu32i.d $t1, %ie64_lo20(foo)
# FIXUP: fixup A - offset: 0, value: %ie64_lo20(foo), kind: FK_NONE

lu52i.d $t1, $t1, %ie64_hi12(foo)
# RELOC: R_LARCH_TLS_IE64_HI12 foo 0x0
# INSTR: lu52i.d $t1, $t1, %ie64_hi12(foo)
# FIXUP: fixup A - offset: 0, value: %ie64_hi12(foo), kind: FK_NONE

pcalau12i $t1, %ld_pc_hi20(foo)
# RELOC: R_LARCH_TLS_LD_PC_HI20 foo 0x0
# INSTR: pcalau12i $t1, %ld_pc_hi20(foo)
# FIXUP: fixup A - offset: 0, value: %ld_pc_hi20(foo), kind: FK_NONE

lu12i.w $t1, %ld_hi20(foo)
# RELOC: R_LARCH_TLS_LD_HI20 foo 0x0
# INSTR: lu12i.w $t1, %ld_hi20(foo)
# FIXUP: fixup A - offset: 0, value: %ld_hi20(foo), kind: FK_NONE

pcalau12i $t1, %gd_pc_hi20(foo)
# RELOC: R_LARCH_TLS_GD_PC_HI20 foo 0x0
# INSTR: pcalau12i $t1, %gd_pc_hi20(foo)
# FIXUP: fixup A - offset: 0, value: %gd_pc_hi20(foo), kind: FK_NONE

lu12i.w $t1, %gd_hi20(foo)
# RELOC: R_LARCH_TLS_GD_HI20 foo 0x0
# INSTR: lu12i.w $t1, %gd_hi20(foo)
# FIXUP: fixup A - offset: 0, value: %gd_hi20(foo), kind: FK_NONE

pcaddu18i $t1, %call36(foo)
# RELOC: R_LARCH_CALL36 foo 0x0
# INSTR: pcaddu18i $t1, %call36(foo)
# FIXUP: fixup A - offset: 0, value: %call36(foo), kind: FK_NONE

pcalau12i $t1, %desc_pc_hi20(foo)
# RELOC: R_LARCH_TLS_DESC_PC_HI20 foo 0x0
# INSTR: pcalau12i $t1, %desc_pc_hi20(foo)
# FIXUP: fixup A - offset: 0, value: %desc_pc_hi20(foo), kind: FK_NONE

addi.d $t1, $t1, %desc_pc_lo12(foo)
# RELOC: R_LARCH_TLS_DESC_PC_LO12 foo 0x0
# INSTR: addi.d $t1, $t1, %desc_pc_lo12(foo)
# FIXUP: fixup A - offset: 0, value: %desc_pc_lo12(foo), kind: FK_NONE

lu32i.d $t1, %desc64_pc_lo20(foo)
# RELOC: R_LARCH_TLS_DESC64_PC_LO20 foo 0x0
# INSTR: lu32i.d $t1, %desc64_pc_lo20(foo)
# FIXUP: fixup A - offset: 0, value: %desc64_pc_lo20(foo), kind: FK_NONE

lu52i.d $t1, $t1, %desc64_pc_hi12(foo)
# RELOC: R_LARCH_TLS_DESC64_PC_HI12 foo 0x0
# INSTR: lu52i.d $t1, $t1, %desc64_pc_hi12(foo)
# FIXUP: fixup A - offset: 0, value: %desc64_pc_hi12(foo), kind: FK_NONE

ld.d $ra, $t1, %desc_ld(foo)
# RELOC: R_LARCH_TLS_DESC_LD foo 0x0
# INSTR: ld.d $ra, $t1, %desc_ld(foo)
# FIXUP: fixup A - offset: 0, value: %desc_ld(foo), kind: FK_NONE

jirl $ra, $ra, %desc_call(foo)
# RELOC: R_LARCH_TLS_DESC_CALL foo 0x0
# INSTR: jirl $ra, $ra, %desc_call(foo)
# FIXUP: fixup A - offset: 0, value: %desc_call(foo), kind: FK_NONE

lu12i.w $t1, %desc_hi20(foo)
# RELOC: R_LARCH_TLS_DESC_HI20 foo 0x0
# INSTR: lu12i.w $t1, %desc_hi20(foo)
# FIXUP: fixup A - offset: 0, value: %desc_hi20(foo), kind: FK_NONE

ori $t1, $t1, %desc_lo12(foo)
# RELOC: R_LARCH_TLS_DESC_LO12 foo 0x0
# INSTR: ori $t1, $t1, %desc_lo12(foo)
# FIXUP: fixup A - offset: 0, value: %desc_lo12(foo), kind: FK_NONE

lu32i.d $t1, %desc64_lo20(foo)
# RELOC: R_LARCH_TLS_DESC64_LO20 foo 0x0
# INSTR: lu32i.d $t1, %desc64_lo20(foo)
# FIXUP: fixup A - offset: 0, value: %desc64_lo20(foo), kind: FK_NONE

lu52i.d $t1, $t1, %desc64_hi12(foo)
# RELOC: R_LARCH_TLS_DESC64_HI12 foo 0x0
# INSTR: lu52i.d $t1, $t1, %desc64_hi12(foo)
# FIXUP: fixup A - offset: 0, value: %desc64_hi12(foo), kind: FK_NONE
