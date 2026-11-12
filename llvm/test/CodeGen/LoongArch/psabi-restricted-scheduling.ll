; RUN: llc --mtriple=loongarch64 -mattr=+d --code-model=medium --relocation-model=pic --post-RA-scheduler=0 < %s \
; RUN:     | FileCheck %s --check-prefix=MEDIUM_NO_SCH
; RUN: llc --mtriple=loongarch64 -mattr=+d --code-model=medium --relocation-model=pic --post-RA-scheduler=1 < %s \
; RUN:     | FileCheck %s --check-prefix=MEDIUM_SCH
; RUN: llc --mtriple=loongarch64 -mattr=+d --code-model=large --relocation-model=pic --post-RA-scheduler=0 < %s \
; RUN:     | FileCheck %s --check-prefix=LARGE_NO_SCH
; RUN: llc --mtriple=loongarch64 -mattr=+d --code-model=large --relocation-model=pic --post-RA-scheduler=1 < %s \
; RUN:     | FileCheck %s --check-prefix=LARGE_SCH
; RUN: llc --mtriple=loongarch64 -mattr=+d --code-model=medium --relocation-model=pic --enable-tlsdesc \
; RUN:     --post-RA-scheduler=0 < %s | FileCheck %s --check-prefix=MEDIUMDESC_NO_SCH
; RUN: llc --mtriple=loongarch64 -mattr=+d --code-model=medium --relocation-model=pic --enable-tlsdesc \
; RUN:     --post-RA-scheduler=1 < %s | FileCheck %s --check-prefix=MEDIUMDESC_SCH
; RUN: llc --mtriple=loongarch64 -mattr=+d --code-model=large --relocation-model=pic --enable-tlsdesc \
; RUN:     --post-RA-scheduler=0 < %s | FileCheck %s --check-prefix=LARGEDESC_NO_SCH
; RUN: llc --mtriple=loongarch64 -mattr=+d --code-model=large --relocation-model=pic --enable-tlsdesc \
; RUN:     --post-RA-scheduler=1 < %s | FileCheck %s --check-prefix=LARGEDESC_SCH

@g = dso_local global i64 zeroinitializer, align 4
@G = global i64 zeroinitializer, align 4
@gd = external thread_local global i64
@ld = external thread_local(localdynamic) global i64
@ie = external thread_local(initialexec) global i64

declare ptr @bar(i64)

define void @foo() nounwind {
; MEDIUM_NO_SCH-LABEL: foo:
; MEDIUM_NO_SCH:       # %bb.0:
; MEDIUM_NO_SCH-NEXT:    addi.d $sp, $sp, -16
; MEDIUM_NO_SCH-NEXT:    st.d $ra, $sp, 8 # 8-byte Folded Spill
; MEDIUM_NO_SCH-NEXT:    pcalau12i $a0, %got_pc_hi20(G)
; MEDIUM_NO_SCH-NEXT:    ld.d $a0, $a0, %got_pc_lo12(G)
; MEDIUM_NO_SCH-NEXT:    ld.d $zero, $a0, 0
; MEDIUM_NO_SCH-NEXT:    pcalau12i $a0, %pc_hi20(.Lg$local)
; MEDIUM_NO_SCH-NEXT:    ld.d $zero, $a0, %pc_lo12(.Lg$local)
; MEDIUM_NO_SCH-NEXT:    ori $a0, $zero, 1
; MEDIUM_NO_SCH-NEXT:    pcaddu18i $ra, %call36(bar)
; MEDIUM_NO_SCH-NEXT:    jirl $ra, $ra, 0
; MEDIUM_NO_SCH-NEXT:    pcalau12i $a0, %gd_pc_hi20(gd)
; MEDIUM_NO_SCH-NEXT:    addi.d $a0, $a0, %got_pc_lo12(gd)
; MEDIUM_NO_SCH-NEXT:    pcaddu18i $ra, %call36(__tls_get_addr)
; MEDIUM_NO_SCH-NEXT:    jirl $ra, $ra, 0
; MEDIUM_NO_SCH-NEXT:    ld.d $zero, $a0, 0
; MEDIUM_NO_SCH-NEXT:    pcalau12i $a0, %ld_pc_hi20(ld)
; MEDIUM_NO_SCH-NEXT:    addi.d $a0, $a0, %got_pc_lo12(ld)
; MEDIUM_NO_SCH-NEXT:    pcaddu18i $ra, %call36(__tls_get_addr)
; MEDIUM_NO_SCH-NEXT:    jirl $ra, $ra, 0
; MEDIUM_NO_SCH-NEXT:    pcalau12i $a1, %ie_pc_hi20(ie)
; MEDIUM_NO_SCH-NEXT:    ld.d $a1, $a1, %ie_pc_lo12(ie)
; MEDIUM_NO_SCH-NEXT:    ld.d $zero, $a0, 0
; MEDIUM_NO_SCH-NEXT:    ldx.d $zero, $a1, $tp
; MEDIUM_NO_SCH-NEXT:    ld.d $ra, $sp, 8 # 8-byte Folded Reload
; MEDIUM_NO_SCH-NEXT:    addi.d $sp, $sp, 16
; MEDIUM_NO_SCH-NEXT:    ret
;
; MEDIUM_SCH-LABEL: foo:
; MEDIUM_SCH:       # %bb.0:
; MEDIUM_SCH-NEXT:    addi.d $sp, $sp, -16
; MEDIUM_SCH-NEXT:    st.d $ra, $sp, 8 # 8-byte Folded Spill
; MEDIUM_SCH-NEXT:    pcalau12i $a0, %got_pc_hi20(G)
; MEDIUM_SCH-NEXT:    ld.d $a0, $a0, %got_pc_lo12(G)
; MEDIUM_SCH-NEXT:    ld.d $zero, $a0, 0
; MEDIUM_SCH-NEXT:    pcalau12i $a0, %pc_hi20(.Lg$local)
; MEDIUM_SCH-NEXT:    ld.d $zero, $a0, %pc_lo12(.Lg$local)
; MEDIUM_SCH-NEXT:    ori $a0, $zero, 1
; MEDIUM_SCH-NEXT:    pcaddu18i $ra, %call36(bar)
; MEDIUM_SCH-NEXT:    jirl $ra, $ra, 0
; MEDIUM_SCH-NEXT:    pcalau12i $a0, %gd_pc_hi20(gd)
; MEDIUM_SCH-NEXT:    addi.d $a0, $a0, %got_pc_lo12(gd)
; MEDIUM_SCH-NEXT:    pcaddu18i $ra, %call36(__tls_get_addr)
; MEDIUM_SCH-NEXT:    jirl $ra, $ra, 0
; MEDIUM_SCH-NEXT:    ld.d $zero, $a0, 0
; MEDIUM_SCH-NEXT:    pcalau12i $a0, %ld_pc_hi20(ld)
; MEDIUM_SCH-NEXT:    addi.d $a0, $a0, %got_pc_lo12(ld)
; MEDIUM_SCH-NEXT:    pcaddu18i $ra, %call36(__tls_get_addr)
; MEDIUM_SCH-NEXT:    jirl $ra, $ra, 0
; MEDIUM_SCH-NEXT:    pcalau12i $a1, %ie_pc_hi20(ie)
; MEDIUM_SCH-NEXT:    ld.d $zero, $a0, 0
; MEDIUM_SCH-NEXT:    ld.d $a1, $a1, %ie_pc_lo12(ie)
; MEDIUM_SCH-NEXT:    ldx.d $zero, $a1, $tp
; MEDIUM_SCH-NEXT:    ld.d $ra, $sp, 8 # 8-byte Folded Reload
; MEDIUM_SCH-NEXT:    addi.d $sp, $sp, 16
; MEDIUM_SCH-NEXT:    ret
;
; LARGE_NO_SCH-LABEL: foo:
; LARGE_NO_SCH:       # %bb.0:
; LARGE_NO_SCH-NEXT:    addi.d $sp, $sp, -16
; LARGE_NO_SCH-NEXT:    st.d $ra, $sp, 8 # 8-byte Folded Spill
; LARGE_NO_SCH-NEXT:    pcalau12i $a0, %got_pc_hi20(G)
; LARGE_NO_SCH-NEXT:    addi.d $a1, $zero, %got_pc_lo12(G)
; LARGE_NO_SCH-NEXT:    lu32i.d $a1, %got64_pc_lo20(G)
; LARGE_NO_SCH-NEXT:    lu52i.d $a1, $a1, %got64_pc_hi12(G)
; LARGE_NO_SCH-NEXT:    ldx.d $a0, $a1, $a0
; LARGE_NO_SCH-NEXT:    ld.d $zero, $a0, 0
; LARGE_NO_SCH-NEXT:    pcalau12i $a0, %pc_hi20(.Lg$local)
; LARGE_NO_SCH-NEXT:    addi.d $a1, $zero, %pc_lo12(.Lg$local)
; LARGE_NO_SCH-NEXT:    lu32i.d $a1, %pc64_lo20(.Lg$local)
; LARGE_NO_SCH-NEXT:    lu52i.d $a1, $a1, %pc64_hi12(.Lg$local)
; LARGE_NO_SCH-NEXT:    ldx.d $zero, $a1, $a0
; LARGE_NO_SCH-NEXT:    ori $a0, $zero, 1
; LARGE_NO_SCH-NEXT:    pcalau12i $a1, %got_pc_hi20(bar)
; LARGE_NO_SCH-NEXT:    addi.d $ra, $zero, %got_pc_lo12(bar)
; LARGE_NO_SCH-NEXT:    lu32i.d $ra, %got64_pc_lo20(bar)
; LARGE_NO_SCH-NEXT:    lu52i.d $ra, $ra, %got64_pc_hi12(bar)
; LARGE_NO_SCH-NEXT:    ldx.d $ra, $ra, $a1
; LARGE_NO_SCH-NEXT:    jirl $ra, $ra, 0
; LARGE_NO_SCH-NEXT:    pcalau12i $a0, %gd_pc_hi20(gd)
; LARGE_NO_SCH-NEXT:    addi.d $a1, $zero, %got_pc_lo12(gd)
; LARGE_NO_SCH-NEXT:    lu32i.d $a1, %got64_pc_lo20(gd)
; LARGE_NO_SCH-NEXT:    lu52i.d $a1, $a1, %got64_pc_hi12(gd)
; LARGE_NO_SCH-NEXT:    add.d $a0, $a1, $a0
; LARGE_NO_SCH-NEXT:    pcalau12i $a1, %got_pc_hi20(__tls_get_addr)
; LARGE_NO_SCH-NEXT:    addi.d $ra, $zero, %got_pc_lo12(__tls_get_addr)
; LARGE_NO_SCH-NEXT:    lu32i.d $ra, %got64_pc_lo20(__tls_get_addr)
; LARGE_NO_SCH-NEXT:    lu52i.d $ra, $ra, %got64_pc_hi12(__tls_get_addr)
; LARGE_NO_SCH-NEXT:    ldx.d $ra, $ra, $a1
; LARGE_NO_SCH-NEXT:    jirl $ra, $ra, 0
; LARGE_NO_SCH-NEXT:    ld.d $zero, $a0, 0
; LARGE_NO_SCH-NEXT:    pcalau12i $a0, %ld_pc_hi20(ld)
; LARGE_NO_SCH-NEXT:    addi.d $a1, $zero, %got_pc_lo12(ld)
; LARGE_NO_SCH-NEXT:    lu32i.d $a1, %got64_pc_lo20(ld)
; LARGE_NO_SCH-NEXT:    lu52i.d $a1, $a1, %got64_pc_hi12(ld)
; LARGE_NO_SCH-NEXT:    add.d $a0, $a1, $a0
; LARGE_NO_SCH-NEXT:    pcalau12i $a1, %got_pc_hi20(__tls_get_addr)
; LARGE_NO_SCH-NEXT:    addi.d $ra, $zero, %got_pc_lo12(__tls_get_addr)
; LARGE_NO_SCH-NEXT:    lu32i.d $ra, %got64_pc_lo20(__tls_get_addr)
; LARGE_NO_SCH-NEXT:    lu52i.d $ra, $ra, %got64_pc_hi12(__tls_get_addr)
; LARGE_NO_SCH-NEXT:    ldx.d $ra, $ra, $a1
; LARGE_NO_SCH-NEXT:    jirl $ra, $ra, 0
; LARGE_NO_SCH-NEXT:    ld.d $zero, $a0, 0
; LARGE_NO_SCH-NEXT:    pcalau12i $a0, %ie_pc_hi20(ie)
; LARGE_NO_SCH-NEXT:    addi.d $a1, $zero, %ie_pc_lo12(ie)
; LARGE_NO_SCH-NEXT:    lu32i.d $a1, %ie64_pc_lo20(ie)
; LARGE_NO_SCH-NEXT:    lu52i.d $a1, $a1, %ie64_pc_hi12(ie)
; LARGE_NO_SCH-NEXT:    ldx.d $a0, $a1, $a0
; LARGE_NO_SCH-NEXT:    ldx.d $zero, $a0, $tp
; LARGE_NO_SCH-NEXT:    ld.d $ra, $sp, 8 # 8-byte Folded Reload
; LARGE_NO_SCH-NEXT:    addi.d $sp, $sp, 16
; LARGE_NO_SCH-NEXT:    ret
;
; LARGE_SCH-LABEL: foo:
; LARGE_SCH:       # %bb.0:
; LARGE_SCH-NEXT:    addi.d $sp, $sp, -16
; LARGE_SCH-NEXT:    st.d $ra, $sp, 8 # 8-byte Folded Spill
; LARGE_SCH-NEXT:    pcalau12i $a0, %got_pc_hi20(G)
; LARGE_SCH-NEXT:    addi.d $a1, $zero, %got_pc_lo12(G)
; LARGE_SCH-NEXT:    lu32i.d $a1, %got64_pc_lo20(G)
; LARGE_SCH-NEXT:    lu52i.d $a1, $a1, %got64_pc_hi12(G)
; LARGE_SCH-NEXT:    ldx.d $a0, $a1, $a0
; LARGE_SCH-NEXT:    ld.d $zero, $a0, 0
; LARGE_SCH-NEXT:    pcalau12i $a0, %pc_hi20(.Lg$local)
; LARGE_SCH-NEXT:    addi.d $a1, $zero, %pc_lo12(.Lg$local)
; LARGE_SCH-NEXT:    lu32i.d $a1, %pc64_lo20(.Lg$local)
; LARGE_SCH-NEXT:    lu52i.d $a1, $a1, %pc64_hi12(.Lg$local)
; LARGE_SCH-NEXT:    ldx.d $zero, $a1, $a0
; LARGE_SCH-NEXT:    ori $a0, $zero, 1
; LARGE_SCH-NEXT:    pcalau12i $a1, %got_pc_hi20(bar)
; LARGE_SCH-NEXT:    addi.d $ra, $zero, %got_pc_lo12(bar)
; LARGE_SCH-NEXT:    lu32i.d $ra, %got64_pc_lo20(bar)
; LARGE_SCH-NEXT:    lu52i.d $ra, $ra, %got64_pc_hi12(bar)
; LARGE_SCH-NEXT:    ldx.d $ra, $ra, $a1
; LARGE_SCH-NEXT:    jirl $ra, $ra, 0
; LARGE_SCH-NEXT:    pcalau12i $a0, %gd_pc_hi20(gd)
; LARGE_SCH-NEXT:    addi.d $a1, $zero, %got_pc_lo12(gd)
; LARGE_SCH-NEXT:    lu32i.d $a1, %got64_pc_lo20(gd)
; LARGE_SCH-NEXT:    lu52i.d $a1, $a1, %got64_pc_hi12(gd)
; LARGE_SCH-NEXT:    add.d $a0, $a1, $a0
; LARGE_SCH-NEXT:    pcalau12i $a1, %got_pc_hi20(__tls_get_addr)
; LARGE_SCH-NEXT:    addi.d $ra, $zero, %got_pc_lo12(__tls_get_addr)
; LARGE_SCH-NEXT:    lu32i.d $ra, %got64_pc_lo20(__tls_get_addr)
; LARGE_SCH-NEXT:    lu52i.d $ra, $ra, %got64_pc_hi12(__tls_get_addr)
; LARGE_SCH-NEXT:    ldx.d $ra, $ra, $a1
; LARGE_SCH-NEXT:    jirl $ra, $ra, 0
; LARGE_SCH-NEXT:    ld.d $zero, $a0, 0
; LARGE_SCH-NEXT:    pcalau12i $a0, %ld_pc_hi20(ld)
; LARGE_SCH-NEXT:    addi.d $a1, $zero, %got_pc_lo12(ld)
; LARGE_SCH-NEXT:    lu32i.d $a1, %got64_pc_lo20(ld)
; LARGE_SCH-NEXT:    lu52i.d $a1, $a1, %got64_pc_hi12(ld)
; LARGE_SCH-NEXT:    add.d $a0, $a1, $a0
; LARGE_SCH-NEXT:    pcalau12i $a1, %got_pc_hi20(__tls_get_addr)
; LARGE_SCH-NEXT:    addi.d $ra, $zero, %got_pc_lo12(__tls_get_addr)
; LARGE_SCH-NEXT:    lu32i.d $ra, %got64_pc_lo20(__tls_get_addr)
; LARGE_SCH-NEXT:    lu52i.d $ra, $ra, %got64_pc_hi12(__tls_get_addr)
; LARGE_SCH-NEXT:    ldx.d $ra, $ra, $a1
; LARGE_SCH-NEXT:    jirl $ra, $ra, 0
; LARGE_SCH-NEXT:    ld.d $zero, $a0, 0
; LARGE_SCH-NEXT:    pcalau12i $a0, %ie_pc_hi20(ie)
; LARGE_SCH-NEXT:    addi.d $a1, $zero, %ie_pc_lo12(ie)
; LARGE_SCH-NEXT:    lu32i.d $a1, %ie64_pc_lo20(ie)
; LARGE_SCH-NEXT:    lu52i.d $a1, $a1, %ie64_pc_hi12(ie)
; LARGE_SCH-NEXT:    ldx.d $a0, $a1, $a0
; LARGE_SCH-NEXT:    ldx.d $zero, $a0, $tp
; LARGE_SCH-NEXT:    ld.d $ra, $sp, 8 # 8-byte Folded Reload
; LARGE_SCH-NEXT:    addi.d $sp, $sp, 16
; LARGE_SCH-NEXT:    ret
  %V = load volatile i64, ptr @G
  %v = load volatile i64, ptr @g
  call void @bar(i64 1)
  %v_gd = load volatile i64, ptr @gd
  %v_ld = load volatile i64, ptr @ld
  %v_ie = load volatile i64, ptr @ie
  ret void
}

define void @baz() nounwind {
; MEDIUMDESC_NO_SCH-LABEL: baz:
; MEDIUMDESC_NO_SCH:       # %bb.0:
; MEDIUMDESC_NO_SCH-NEXT:    addi.d $sp, $sp, -16
; MEDIUMDESC_NO_SCH-NEXT:    st.d $ra, $sp, 8 # 8-byte Folded Spill
; MEDIUMDESC_NO_SCH-NEXT:    pcalau12i $a0, %desc_pc_hi20(gd)
; MEDIUMDESC_NO_SCH-NEXT:    addi.d $a0, $a0, %desc_pc_lo12(gd)
; MEDIUMDESC_NO_SCH-NEXT:    ld.d $ra, $a0, %desc_ld(gd)
; MEDIUMDESC_NO_SCH-NEXT:    jirl $ra, $ra, %desc_call(gd)
; MEDIUMDESC_NO_SCH-NEXT:    add.d $a0, $a0, $tp
; MEDIUMDESC_NO_SCH-NEXT:    ld.d $zero, $a0, 0
; MEDIUMDESC_NO_SCH-NEXT:    ld.d $ra, $sp, 8 # 8-byte Folded Reload
; MEDIUMDESC_NO_SCH-NEXT:    addi.d $sp, $sp, 16
; MEDIUMDESC_NO_SCH-NEXT:    ret
;
; MEDIUMDESC_SCH-LABEL: baz:
; MEDIUMDESC_SCH:       # %bb.0:
; MEDIUMDESC_SCH-NEXT:    addi.d $sp, $sp, -16
; MEDIUMDESC_SCH-NEXT:    st.d $ra, $sp, 8 # 8-byte Folded Spill
; MEDIUMDESC_SCH-NEXT:    pcalau12i $a0, %desc_pc_hi20(gd)
; MEDIUMDESC_SCH-NEXT:    addi.d $a0, $a0, %desc_pc_lo12(gd)
; MEDIUMDESC_SCH-NEXT:    ld.d $ra, $a0, %desc_ld(gd)
; MEDIUMDESC_SCH-NEXT:    jirl $ra, $ra, %desc_call(gd)
; MEDIUMDESC_SCH-NEXT:    add.d $a0, $a0, $tp
; MEDIUMDESC_SCH-NEXT:    ld.d $zero, $a0, 0
; MEDIUMDESC_SCH-NEXT:    ld.d $ra, $sp, 8 # 8-byte Folded Reload
; MEDIUMDESC_SCH-NEXT:    addi.d $sp, $sp, 16
; MEDIUMDESC_SCH-NEXT:    ret
;
; LARGEDESC_NO_SCH-LABEL: baz:
; LARGEDESC_NO_SCH:       # %bb.0:
; LARGEDESC_NO_SCH-NEXT:    addi.d $sp, $sp, -16
; LARGEDESC_NO_SCH-NEXT:    st.d $ra, $sp, 8 # 8-byte Folded Spill
; LARGEDESC_NO_SCH-NEXT:    pcalau12i $a0, %desc_pc_hi20(gd)
; LARGEDESC_NO_SCH-NEXT:    addi.d $a1, $zero, %desc_pc_lo12(gd)
; LARGEDESC_NO_SCH-NEXT:    lu32i.d $a1, %desc64_pc_lo20(gd)
; LARGEDESC_NO_SCH-NEXT:    lu52i.d $a1, $a1, %desc64_pc_hi12(gd)
; LARGEDESC_NO_SCH-NEXT:    add.d $a0, $a0, $a1
; LARGEDESC_NO_SCH-NEXT:    ld.d $ra, $a0, %desc_ld(gd)
; LARGEDESC_NO_SCH-NEXT:    jirl $ra, $ra, %desc_call(gd)
; LARGEDESC_NO_SCH-NEXT:    add.d $a0, $a0, $tp
; LARGEDESC_NO_SCH-NEXT:    ld.d $zero, $a0, 0
; LARGEDESC_NO_SCH-NEXT:    ld.d $ra, $sp, 8 # 8-byte Folded Reload
; LARGEDESC_NO_SCH-NEXT:    addi.d $sp, $sp, 16
; LARGEDESC_NO_SCH-NEXT:    ret
;
; LARGEDESC_SCH-LABEL: baz:
; LARGEDESC_SCH:       # %bb.0:
; LARGEDESC_SCH-NEXT:    addi.d $sp, $sp, -16
; LARGEDESC_SCH-NEXT:    st.d $ra, $sp, 8 # 8-byte Folded Spill
; LARGEDESC_SCH-NEXT:    pcalau12i $a0, %desc_pc_hi20(gd)
; LARGEDESC_SCH-NEXT:    addi.d $a1, $zero, %desc_pc_lo12(gd)
; LARGEDESC_SCH-NEXT:    lu32i.d $a1, %desc64_pc_lo20(gd)
; LARGEDESC_SCH-NEXT:    lu52i.d $a1, $a1, %desc64_pc_hi12(gd)
; LARGEDESC_SCH-NEXT:    add.d $a0, $a0, $a1
; LARGEDESC_SCH-NEXT:    ld.d $ra, $a0, %desc_ld(gd)
; LARGEDESC_SCH-NEXT:    jirl $ra, $ra, %desc_call(gd)
; LARGEDESC_SCH-NEXT:    add.d $a0, $a0, $tp
; LARGEDESC_SCH-NEXT:    ld.d $zero, $a0, 0
; LARGEDESC_SCH-NEXT:    ld.d $ra, $sp, 8 # 8-byte Folded Reload
; LARGEDESC_SCH-NEXT:    addi.d $sp, $sp, 16
; LARGEDESC_SCH-NEXT:    ret
  %v_gd = load volatile i64, ptr @gd
  ret void
}
