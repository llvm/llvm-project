# RUN: llvm-mc --triple=loongarch64 %s | FileCheck %s

la.abs $a0, sym_abs
# CHECK:      lu12i.w $a0, %abs_hi20(sym_abs)
# CHECK-NEXT: ori $a0, $a0, %abs_lo12(sym_abs)
# CHECK-NEXT: lu32i.d $a0, %abs64_lo20(sym_abs)
# CHECK-NEXT: lu52i.d $a0, $a0, %abs64_hi12(sym_abs)

la.pcrel $a0, sym_pcrel
# CHECK:      pcalau12i $a0, %pc_hi20(sym_pcrel)
# CHECK-NEXT: addi.d $a0, $a0, %pc_lo12(sym_pcrel)

la.pcrel $a0, $a1, sym_pcrel_large
# CHECK:      pcalau12i $a0, %pc_hi20(sym_pcrel_large)
# CHECK-NEXT: addi.d $a1, $zero, %pc_lo12(sym_pcrel_large)
# CHECK-NEXT: lu32i.d $a1, %pc64_lo20(sym_pcrel_large)
# CHECK-NEXT: lu52i.d $a1, $a1, %pc64_hi12(sym_pcrel_large)
# CHECK-NEXT: add.d $a0, $a0, $a1

la.got $a0, sym_got
# CHECK:      pcalau12i $a0, %got_pc_hi20(sym_got)
# CHECK-NEXT: ld.d $a0, $a0, %got_pc_lo12(sym_got)

la.got $a0, $a1, sym_got_large
# CHECK:      pcalau12i $a0, %got_pc_hi20(sym_got_large)
# CHECK-NEXT: addi.d $a1, $zero, %got_pc_lo12(sym_got_large)
# CHECK-NEXT: lu32i.d $a1, %got64_pc_lo20(sym_got_large)
# CHECK-NEXT: lu52i.d $a1, $a1, %got64_pc_hi12(sym_got_large)
# CHECK-NEXT: ldx.d $a0, $a0, $a1

la.tls.le $a0, sym_le
# CHECK:      lu12i.w $a0, %le_hi20(sym_le)
# CHECK-NEXT: ori $a0, $a0, %le_lo12(sym_le)

la.tls.ie $a0, sym_ie
# CHECK:      pcalau12i $a0, %ie_pc_hi20(sym_ie)
# CHECK-NEXT: ld.d $a0, $a0, %ie_pc_lo12(sym_ie)

la.tls.ie $a0, $a1, sym_ie_large
# CHECK:      pcalau12i $a0, %ie_pc_hi20(sym_ie_large)
# CHECK-NEXT: addi.d $a1, $zero, %ie_pc_lo12(sym_ie_large)
# CHECK-NEXT: lu32i.d $a1, %ie64_pc_lo20(sym_ie_large)
# CHECK-NEXT: lu52i.d $a1, $a1, %ie64_pc_hi12(sym_ie_large)
# CHECK-NEXT: ldx.d $a0, $a0, $a1

la.tls.ld $a0, sym_ld
# CHECK:      pcalau12i $a0, %ld_pc_hi20(sym_ld)
# CHECK-NEXT: addi.d $a0, $a0, %got_pc_lo12(sym_ld)

la.tls.ld $a0, $a1, sym_ld_large
# CHECK:      pcalau12i $a0, %ld_pc_hi20(sym_ld_large)
# CHECK-NEXT: addi.d $a1, $zero, %got_pc_lo12(sym_ld_large)
# CHECK-NEXT: lu32i.d $a1, %got64_pc_lo20(sym_ld_large)
# CHECK-NEXT: lu52i.d $a1, $a1, %got64_pc_hi12(sym_ld_large)
# CHECK-NEXT: add.d $a0, $a0, $a1

la.tls.gd $a0, sym_gd
# CHECK:      pcalau12i $a0, %gd_pc_hi20(sym_gd)
# CHECK-NEXT: addi.d $a0, $a0, %got_pc_lo12(sym_gd)

la.tls.gd $a0, $a1, sym_gd_large
# CHECK:      pcalau12i $a0, %gd_pc_hi20(sym_gd_large)
# CHECK-NEXT: addi.d $a1, $zero, %got_pc_lo12(sym_gd_large)
# CHECK-NEXT: lu32i.d $a1, %got64_pc_lo20(sym_gd_large)
# CHECK-NEXT: lu52i.d $a1, $a1, %got64_pc_hi12(sym_gd_large)
# CHECK-NEXT: add.d $a0, $a0, $a1
