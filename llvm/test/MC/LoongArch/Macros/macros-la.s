# RUN: llvm-mc --triple=loongarch64 %s | FileCheck %s
# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=-relax %s -o %t
# RUN: llvm-readobj -r %t | FileCheck %s --check-prefix=RELOC
# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=+relax %s -o %t.relax
# RUN: llvm-readobj -r %t.relax | FileCheck %s --check-prefixes=RELOC,RELAX
# RUN: llvm-mc --triple=loongarch64 --defsym ABS=1 --mattr=+la-global-with-abs \
# RUN:     %s | FileCheck %s --check-prefix=ABS

# RELOC:      Relocations [
# RELOC-NEXT:   Section ({{.*}}) .rela.text {

la.abs $a0, sym_abs
# CHECK:      lu12i.w $a0, %abs_hi20(sym_abs)
# CHECK-NEXT: ori $a0, $a0, %abs_lo12(sym_abs)
# CHECK-NEXT: lu32i.d $a0, %abs64_lo20(sym_abs)
# CHECK-NEXT: lu52i.d $a0, $a0, %abs64_hi12(sym_abs)
# CHECK-EMPTY:
# RELOC-NEXT: R_LARCH_ABS_HI20 sym_abs 0x0
# RELOC-NEXT: R_LARCH_ABS_LO12 sym_abs 0x0
# RELOC-NEXT: R_LARCH_ABS64_LO20 sym_abs 0x0
# RELOC-NEXT: R_LARCH_ABS64_HI12 sym_abs 0x0

la.pcrel $a0, sym_pcrel
# CHECK-NEXT: pcalau12i $a0, %pc_hi20(sym_pcrel)
# CHECK-NEXT: addi.d $a0, $a0, %pc_lo12(sym_pcrel)
# CHECK-EMPTY:
# RELOC-NEXT: R_LARCH_PCALA_HI20 sym_pcrel 0x0
# RELAX-NEXT: R_LARCH_RELAX - 0x0
# RELOC-NEXT: R_LARCH_PCALA_LO12 sym_pcrel 0x0
# RELAX-NEXT: R_LARCH_RELAX - 0x0

la.pcrel $a0, $a1, sym_pcrel_large
# CHECK-NEXT: pcalau12i $a0, %pc_hi20(sym_pcrel_large)
# CHECK-NEXT: addi.d $a1, $zero, %pc_lo12(sym_pcrel_large)
# CHECK-NEXT: lu32i.d $a1, %pc64_lo20(sym_pcrel_large)
# CHECK-NEXT: lu52i.d $a1, $a1, %pc64_hi12(sym_pcrel_large)
# CHECK-NEXT: add.d $a0, $a0, $a1
# CHECK-EMPTY:
# RELOC-NEXT: R_LARCH_PCALA_HI20 sym_pcrel_large 0x0
# RELOC-NEXT: R_LARCH_PCALA_LO12 sym_pcrel_large 0x0
# RELOC-NEXT: R_LARCH_PCALA64_LO20 sym_pcrel_large 0x0
# RELOC-NEXT: R_LARCH_PCALA64_HI12 sym_pcrel_large 0x0

la.got $a0, sym_got
# CHECK-NEXT: pcalau12i $a0, %got_pc_hi20(sym_got)
# CHECK-NEXT: ld.d $a0, $a0, %got_pc_lo12(sym_got)
# CHECK-EMPTY:
# RELOC-NEXT: R_LARCH_GOT_PC_HI20 sym_got 0x0
# RELAX-NEXT: R_LARCH_RELAX - 0x0
# RELOC-NEXT: R_LARCH_GOT_PC_LO12 sym_got 0x0
# RELAX-NEXT: R_LARCH_RELAX - 0x0

la.got $a0, $a1, sym_got_large
# CHECK-NEXT: pcalau12i $a0, %got_pc_hi20(sym_got_large)
# CHECK-NEXT: addi.d $a1, $zero, %got_pc_lo12(sym_got_large)
# CHECK-NEXT: lu32i.d $a1, %got64_pc_lo20(sym_got_large)
# CHECK-NEXT: lu52i.d $a1, $a1, %got64_pc_hi12(sym_got_large)
# CHECK-NEXT: ldx.d $a0, $a0, $a1
# CHECK-EMPTY:
# RELOC-NEXT: R_LARCH_GOT_PC_HI20 sym_got_large 0x0
# RELOC-NEXT: R_LARCH_GOT_PC_LO12 sym_got_large 0x0
# RELOC-NEXT: R_LARCH_GOT64_PC_LO20 sym_got_large 0x0
# RELOC-NEXT: R_LARCH_GOT64_PC_HI12 sym_got_large 0x0

la.tls.le $a0, sym_le
# CHECK-NEXT: lu12i.w $a0, %le_hi20(sym_le)
# CHECK-NEXT: ori $a0, $a0, %le_lo12(sym_le)
# CHECK-EMPTY:
# RELOC-NEXT: R_LARCH_TLS_LE_HI20 sym_le 0x0
# RELOC-NEXT: R_LARCH_TLS_LE_LO12 sym_le 0x0

la.tls.ie $a0, sym_ie
# CHECK-NEXT: pcalau12i $a0, %ie_pc_hi20(sym_ie)
# CHECK-NEXT: ld.d $a0, $a0, %ie_pc_lo12(sym_ie)
# CHECK-EMPTY:
# RELOC-NEXT: R_LARCH_TLS_IE_PC_HI20 sym_ie 0x0
# RELOC-NEXT: R_LARCH_TLS_IE_PC_LO12 sym_ie 0x0

la.tls.ie $a0, $a1, sym_ie_large
# CHECK-NEXT: pcalau12i $a0, %ie_pc_hi20(sym_ie_large)
# CHECK-NEXT: addi.d $a1, $zero, %ie_pc_lo12(sym_ie_large)
# CHECK-NEXT: lu32i.d $a1, %ie64_pc_lo20(sym_ie_large)
# CHECK-NEXT: lu52i.d $a1, $a1, %ie64_pc_hi12(sym_ie_large)
# CHECK-NEXT: ldx.d $a0, $a0, $a1
# CHECK-EMPTY:
# RELOC-NEXT: R_LARCH_TLS_IE_PC_HI20 sym_ie_large 0x0
# RELOC-NEXT: R_LARCH_TLS_IE_PC_LO12 sym_ie_large 0x0
# RELOC-NEXT: R_LARCH_TLS_IE64_PC_LO20 sym_ie_large 0x0
# RELOC-NEXT: R_LARCH_TLS_IE64_PC_HI12 sym_ie_large 0x0

la.tls.ld $a0, sym_ld
# CHECK-NEXT: pcalau12i $a0, %ld_pc_hi20(sym_ld)
# CHECK-NEXT: addi.d $a0, $a0, %got_pc_lo12(sym_ld)
# CHECK-EMPTY:
# RELOC-NEXT: R_LARCH_TLS_LD_PC_HI20 sym_ld 0x0
# RELOC-NEXT: R_LARCH_GOT_PC_LO12 sym_ld 0x0

la.tls.ld $a0, $a1, sym_ld_large
# CHECK-NEXT: pcalau12i $a0, %ld_pc_hi20(sym_ld_large)
# CHECK-NEXT: addi.d $a1, $zero, %got_pc_lo12(sym_ld_large)
# CHECK-NEXT: lu32i.d $a1, %got64_pc_lo20(sym_ld_large)
# CHECK-NEXT: lu52i.d $a1, $a1, %got64_pc_hi12(sym_ld_large)
# CHECK-NEXT: add.d $a0, $a0, $a1
# CHECK-EMPTY:
# RELOC-NEXT: R_LARCH_TLS_LD_PC_HI20 sym_ld_large 0x0
# RELOC-NEXT: R_LARCH_GOT_PC_LO12 sym_ld_large 0x0
# RELOC-NEXT: R_LARCH_GOT64_PC_LO20 sym_ld_large 0x0
# RELOC-NEXT: R_LARCH_GOT64_PC_HI12 sym_ld_large 0x0

la.tls.gd $a0, sym_gd
# CHECK-NEXT: pcalau12i $a0, %gd_pc_hi20(sym_gd)
# CHECK-NEXT: addi.d $a0, $a0, %got_pc_lo12(sym_gd)
# CHECK-EMPTY:
# RELOC-NEXT: R_LARCH_TLS_GD_PC_HI20 sym_gd 0x0
# RELOC-NEXT: R_LARCH_GOT_PC_LO12 sym_gd 0x0

la.tls.gd $a0, $a1, sym_gd_large
# CHECK-NEXT: pcalau12i $a0, %gd_pc_hi20(sym_gd_large)
# CHECK-NEXT: addi.d $a1, $zero, %got_pc_lo12(sym_gd_large)
# CHECK-NEXT: lu32i.d $a1, %got64_pc_lo20(sym_gd_large)
# CHECK-NEXT: lu52i.d $a1, $a1, %got64_pc_hi12(sym_gd_large)
# CHECK-NEXT: add.d $a0, $a0, $a1
# CHECK-EMPTY:
# RELOC-NEXT: R_LARCH_TLS_GD_PC_HI20 sym_gd_large 0x0
# RELOC-NEXT: R_LARCH_GOT_PC_LO12 sym_gd_large 0x0
# RELOC-NEXT: R_LARCH_GOT64_PC_LO20 sym_gd_large 0x0
# RELOC-NEXT: R_LARCH_GOT64_PC_HI12 sym_gd_large 0x0

la.tls.desc $a0, sym_desc
# CHECK-NEXT: pcalau12i $a0, %desc_pc_hi20(sym_desc)
# CHECK-NEXT: addi.d $a0, $a0, %desc_pc_lo12(sym_desc)
# CHECK-NEXT: ld.d $ra, $a0, %desc_ld(sym_desc)
# CHECK-NEXT: jirl $ra, $ra, %desc_call(sym_desc)
# CHECK-EMPTY:
# RELOC-NEXT: R_LARCH_TLS_DESC_PC_HI20 sym_desc 0x0
# RELOC-NEXT: R_LARCH_TLS_DESC_PC_LO12 sym_desc 0x0
# RELOC-NEXT: R_LARCH_TLS_DESC_LD sym_desc 0x0
# RELOC-NEXT: R_LARCH_TLS_DESC_CALL sym_desc 0x0

la.tls.desc $a0, $a1, sym_desc_large
# CHECK-NEXT: pcalau12i $a0, %desc_pc_hi20(sym_desc_large)
# CHECK-NEXT: addi.d $a1, $zero, %desc_pc_lo12(sym_desc_large)
# CHECK-NEXT: lu32i.d $a1, %desc64_pc_lo20(sym_desc_large)
# CHECK-NEXT: lu52i.d $a1, $a1, %desc64_pc_hi12(sym_desc_large)
# CHECK-NEXT: add.d $a0, $a0, $a1
# CHECK-NEXT: ld.d $ra, $a0, %desc_ld(sym_desc_large)
# CHECK-NEXT: jirl $ra, $ra, %desc_call(sym_desc_large)
# CHECK-EMPTY:
# RELOC-NEXT: R_LARCH_TLS_DESC_PC_HI20 sym_desc_large 0x0
# RELOC-NEXT: R_LARCH_TLS_DESC_PC_LO12 sym_desc_large 0x0
# RELOC-NEXT: R_LARCH_TLS_DESC64_PC_LO20 sym_desc_large 0x0
# RELOC-NEXT: R_LARCH_TLS_DESC64_PC_HI12 sym_desc_large 0x0
# RELOC-NEXT: R_LARCH_TLS_DESC_LD sym_desc_large 0x0
# RELOC-NEXT: R_LARCH_TLS_DESC_CALL sym_desc_large 0x0


# RELOC-NEXT:   }
# RELOC-NEXT: ]

#############################################################
## with feature: +la-global-with-abs
#############################################################
.ifdef ABS

la.tls.desc $a0, sym_desc
# ABS:      lu12i.w $a0, %desc_hi20(sym_desc)
# ABS-NEXT: ori $a0, $a0, %desc_lo12(sym_desc)
# ABS-NEXT: lu32i.d $a0, %desc64_lo20(sym_desc)
# ABS-NEXT: lu52i.d $a0, $a0, %desc64_hi12(sym_desc)
# ABS-NEXT: ld.d $ra, $a0, %desc_ld(sym_desc)
# ABS-NEXT: jirl $ra, $ra, %desc_call(sym_desc)

.endif
