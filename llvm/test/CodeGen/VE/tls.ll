; FIXME: even under non-pic mode, llvm needs to generate pic code since nld
;        doesn't work with non-pic code.  Thefore, we test pic codes for
;        both cases here.
;      llc -mtriple ve < %s | FileCheck %s -check-prefix=LOCAL
; RUN: llc -mtriple ve < %s | FileCheck %s -check-prefix=GENDYN
; RUN: llc -mtriple ve -relocation-model=pic < %s | FileCheck %s -check-prefix=GENDYNPIC

@x = external thread_local global i32, align 4
@y = internal thread_local global i32 0, align 4

; Function Attrs: norecurse nounwind readnone
define nonnull i32* @get_global() {
; GENDYN-LABEL: get_global:
; GENDYN:       # %bb.0: # %entry
; GENDYN-NEXT:    st %s9, (,%s11)
; GENDYN-NEXT:    st %s10, 8(,%s11)
; GENDYN-NEXT:    st %s15, 24(,%s11)
; GENDYN-NEXT:    st %s16, 32(,%s11)
; GENDYN-NEXT:    or %s9, 0, %s11
; GENDYN-NEXT:    lea %s13, -240
; GENDYN-NEXT:    and %s13, %s13, (32)0
; GENDYN-NEXT:    lea.sl %s11, -1(%s11, %s13)
; GENDYN-NEXT:    brge.l %s11, %s8, .LBB0_2
; GENDYN-NEXT:  # %bb.1: # %entry
; GENDYN-NEXT:    ld %s61, 24(,%s14)
; GENDYN-NEXT:    or %s62, 0, %s0
; GENDYN-NEXT:    lea %s63, 315
; GENDYN-NEXT:    shm.l %s63, (%s61)
; GENDYN-NEXT:    shm.l %s8, 8(%s61)
; GENDYN-NEXT:    shm.l %s11, 16(%s61)
; GENDYN-NEXT:    monc
; GENDYN-NEXT:    or %s0, 0, %s62
; GENDYN-NEXT:  .LBB0_2: # %entry
; GENDYN-NEXT:    lea %s0, x@tls_gd_lo(-24)
; GENDYN-NEXT:    and %s0, %s0, (32)0
; GENDYN-NEXT:    sic %s10
; GENDYN-NEXT:    lea.sl %s0, x@tls_gd_hi(%s10, %s0)
; GENDYN-NEXT:    lea %s12, __tls_get_addr@plt_lo(8)
; GENDYN-NEXT:    and %s12, %s12, (32)0
; GENDYN-NEXT:    lea.sl %s12, __tls_get_addr@plt_hi(%s10, %s12)
; GENDYN-NEXT:    bsic %s10, (, %s12)
; GENDYN-NEXT:    or %s11, 0, %s9
; GENDYN-NEXT:    ld %s16, 32(,%s11)
; GENDYN-NEXT:    ld %s15, 24(,%s11)
; GENDYN-NEXT:    ld %s10, 8(,%s11)
; GENDYN-NEXT:    ld %s9, (,%s11)
; GENDYN-NEXT:    b.l (,%lr)
;
; GENDYNPIC-LABEL: get_global:
; GENDYNPIC:       # %bb.0: # %entry
; GENDYNPIC-NEXT:    st %s9, (,%s11)
; GENDYNPIC-NEXT:    st %s10, 8(,%s11)
; GENDYNPIC-NEXT:    st %s15, 24(,%s11)
; GENDYNPIC-NEXT:    st %s16, 32(,%s11)
; GENDYNPIC-NEXT:    or %s9, 0, %s11
; GENDYNPIC-NEXT:    lea %s13, -240
; GENDYNPIC-NEXT:    and %s13, %s13, (32)0
; GENDYNPIC-NEXT:    lea.sl %s11, -1(%s11, %s13)
; GENDYNPIC-NEXT:    brge.l %s11, %s8, .LBB0_2
; GENDYNPIC-NEXT:  # %bb.1: # %entry
; GENDYNPIC-NEXT:    ld %s61, 24(,%s14)
; GENDYNPIC-NEXT:    or %s62, 0, %s0
; GENDYNPIC-NEXT:    lea %s63, 315
; GENDYNPIC-NEXT:    shm.l %s63, (%s61)
; GENDYNPIC-NEXT:    shm.l %s8, 8(%s61)
; GENDYNPIC-NEXT:    shm.l %s11, 16(%s61)
; GENDYNPIC-NEXT:    monc
; GENDYNPIC-NEXT:    or %s0, 0, %s62
; GENDYNPIC-NEXT:  .LBB0_2: # %entry
; GENDYNPIC-NEXT:    lea %s15, _GLOBAL_OFFSET_TABLE_@pc_lo(-24)
; GENDYNPIC-NEXT:    and %s15, %s15, (32)0
; GENDYNPIC-NEXT:    sic %s16
; GENDYNPIC-NEXT:    lea.sl %s15, _GLOBAL_OFFSET_TABLE_@pc_hi(%s16, %s15)
; GENDYNPIC-NEXT:    lea %s0, x@tls_gd_lo(-24)
; GENDYNPIC-NEXT:    and %s0, %s0, (32)0
; GENDYNPIC-NEXT:    sic %s10
; GENDYNPIC-NEXT:    lea.sl %s0, x@tls_gd_hi(%s10, %s0)
; GENDYNPIC-NEXT:    lea %s12, __tls_get_addr@plt_lo(8)
; GENDYNPIC-NEXT:    and %s12, %s12, (32)0
; GENDYNPIC-NEXT:    lea.sl %s12, __tls_get_addr@plt_hi(%s10, %s12)
; GENDYNPIC-NEXT:    bsic %s10, (, %s12)
; GENDYNPIC-NEXT:    or %s11, 0, %s9
; GENDYNPIC-NEXT:    ld %s16, 32(,%s11)
; GENDYNPIC-NEXT:    ld %s15, 24(,%s11)
; GENDYNPIC-NEXT:    ld %s10, 8(,%s11)
; GENDYNPIC-NEXT:    ld %s9, (,%s11)
; GENDYNPIC-NEXT:    b.l (,%lr)
; LOCAL-LABEL: get_global:
; LOCAL:       .LBB{{[0-9]+}}_2:
; LOCAL-NEXT:  lea %s34, x@tpoff_lo
; LOCAL-NEXT:  and %s34, %s34, (32)0
; LOCAL-NEXT:  lea.sl %s34, x@tpoff_hi(%s34)
; LOCAL-NEXT:  adds.l %s0, %s14, %s34
; LOCAL-NEXT:  or %s11, 0, %s9
entry:
  ret i32* @x
}

; Function Attrs: norecurse nounwind readnone
define nonnull i32* @get_local() {
; GENDYN-LABEL: get_local:
; GENDYN:       # %bb.0: # %entry
; GENDYN-NEXT:    st %s9, (,%s11)
; GENDYN-NEXT:    st %s10, 8(,%s11)
; GENDYN-NEXT:    st %s15, 24(,%s11)
; GENDYN-NEXT:    st %s16, 32(,%s11)
; GENDYN-NEXT:    or %s9, 0, %s11
; GENDYN-NEXT:    lea %s13, -240
; GENDYN-NEXT:    and %s13, %s13, (32)0
; GENDYN-NEXT:    lea.sl %s11, -1(%s11, %s13)
; GENDYN-NEXT:    brge.l %s11, %s8, .LBB1_2
; GENDYN-NEXT:  # %bb.1: # %entry
; GENDYN-NEXT:    ld %s61, 24(,%s14)
; GENDYN-NEXT:    or %s62, 0, %s0
; GENDYN-NEXT:    lea %s63, 315
; GENDYN-NEXT:    shm.l %s63, (%s61)
; GENDYN-NEXT:    shm.l %s8, 8(%s61)
; GENDYN-NEXT:    shm.l %s11, 16(%s61)
; GENDYN-NEXT:    monc
; GENDYN-NEXT:    or %s0, 0, %s62
; GENDYN-NEXT:  .LBB1_2: # %entry
; GENDYN-NEXT:    lea %s0, y@tls_gd_lo(-24)
; GENDYN-NEXT:    and %s0, %s0, (32)0
; GENDYN-NEXT:    sic %s10
; GENDYN-NEXT:    lea.sl %s0, y@tls_gd_hi(%s10, %s0)
; GENDYN-NEXT:    lea %s12, __tls_get_addr@plt_lo(8)
; GENDYN-NEXT:    and %s12, %s12, (32)0
; GENDYN-NEXT:    lea.sl %s12, __tls_get_addr@plt_hi(%s10, %s12)
; GENDYN-NEXT:    bsic %s10, (, %s12)
; GENDYN-NEXT:    or %s11, 0, %s9
; GENDYN-NEXT:    ld %s16, 32(,%s11)
; GENDYN-NEXT:    ld %s15, 24(,%s11)
; GENDYN-NEXT:    ld %s10, 8(,%s11)
; GENDYN-NEXT:    ld %s9, (,%s11)
; GENDYN-NEXT:    b.l (,%lr)
;
; GENDYNPIC-LABEL: get_local:
; GENDYNPIC:       # %bb.0: # %entry
; GENDYNPIC-NEXT:    st %s9, (,%s11)
; GENDYNPIC-NEXT:    st %s10, 8(,%s11)
; GENDYNPIC-NEXT:    st %s15, 24(,%s11)
; GENDYNPIC-NEXT:    st %s16, 32(,%s11)
; GENDYNPIC-NEXT:    or %s9, 0, %s11
; GENDYNPIC-NEXT:    lea %s13, -240
; GENDYNPIC-NEXT:    and %s13, %s13, (32)0
; GENDYNPIC-NEXT:    lea.sl %s11, -1(%s11, %s13)
; GENDYNPIC-NEXT:    brge.l %s11, %s8, .LBB1_2
; GENDYNPIC-NEXT:  # %bb.1: # %entry
; GENDYNPIC-NEXT:    ld %s61, 24(,%s14)
; GENDYNPIC-NEXT:    or %s62, 0, %s0
; GENDYNPIC-NEXT:    lea %s63, 315
; GENDYNPIC-NEXT:    shm.l %s63, (%s61)
; GENDYNPIC-NEXT:    shm.l %s8, 8(%s61)
; GENDYNPIC-NEXT:    shm.l %s11, 16(%s61)
; GENDYNPIC-NEXT:    monc
; GENDYNPIC-NEXT:    or %s0, 0, %s62
; GENDYNPIC-NEXT:  .LBB1_2: # %entry
; GENDYNPIC-NEXT:    lea %s15, _GLOBAL_OFFSET_TABLE_@pc_lo(-24)
; GENDYNPIC-NEXT:    and %s15, %s15, (32)0
; GENDYNPIC-NEXT:    sic %s16
; GENDYNPIC-NEXT:    lea.sl %s15, _GLOBAL_OFFSET_TABLE_@pc_hi(%s16, %s15)
; GENDYNPIC-NEXT:    lea %s0, y@tls_gd_lo(-24)
; GENDYNPIC-NEXT:    and %s0, %s0, (32)0
; GENDYNPIC-NEXT:    sic %s10
; GENDYNPIC-NEXT:    lea.sl %s0, y@tls_gd_hi(%s10, %s0)
; GENDYNPIC-NEXT:    lea %s12, __tls_get_addr@plt_lo(8)
; GENDYNPIC-NEXT:    and %s12, %s12, (32)0
; GENDYNPIC-NEXT:    lea.sl %s12, __tls_get_addr@plt_hi(%s10, %s12)
; GENDYNPIC-NEXT:    bsic %s10, (, %s12)
; GENDYNPIC-NEXT:    or %s11, 0, %s9
; GENDYNPIC-NEXT:    ld %s16, 32(,%s11)
; GENDYNPIC-NEXT:    ld %s15, 24(,%s11)
; GENDYNPIC-NEXT:    ld %s10, 8(,%s11)
; GENDYNPIC-NEXT:    ld %s9, (,%s11)
; GENDYNPIC-NEXT:    b.l (,%lr)
; LOCAL-LABEL: get_local:
; LOCAL:       .LBB{{[0-9]+}}_2:
; LOCAL-NEXT:  lea %s34, y@tpoff_lo
; LOCAL-NEXT:  and %s34, %s34, (32)0
; LOCAL-NEXT:  lea.sl %s34, y@tpoff_hi(%s34)
; LOCAL-NEXT:  adds.l %s0, %s14, %s34
; LOCAL-NEXT:  or %s11, 0, %s9
entry:
  ret i32* @y
}

; Function Attrs: norecurse nounwind
define void @set_global(i32 %v) {
; GENDYN-LABEL: set_global:
; GENDYN:       # %bb.0: # %entry
; GENDYN-NEXT:    st %s9, (,%s11)
; GENDYN-NEXT:    st %s10, 8(,%s11)
; GENDYN-NEXT:    st %s15, 24(,%s11)
; GENDYN-NEXT:    st %s16, 32(,%s11)
; GENDYN-NEXT:    or %s9, 0, %s11
; GENDYN-NEXT:    lea %s13, -240
; GENDYN-NEXT:    and %s13, %s13, (32)0
; GENDYN-NEXT:    lea.sl %s11, -1(%s11, %s13)
; GENDYN-NEXT:    brge.l %s11, %s8, .LBB2_2
; GENDYN-NEXT:  # %bb.1: # %entry
; GENDYN-NEXT:    ld %s61, 24(,%s14)
; GENDYN-NEXT:    or %s62, 0, %s0
; GENDYN-NEXT:    lea %s63, 315
; GENDYN-NEXT:    shm.l %s63, (%s61)
; GENDYN-NEXT:    shm.l %s8, 8(%s61)
; GENDYN-NEXT:    shm.l %s11, 16(%s61)
; GENDYN-NEXT:    monc
; GENDYN-NEXT:    or %s0, 0, %s62
; GENDYN-NEXT:  .LBB2_2: # %entry
; GENDYN-NEXT:    st %s18, 48(,%s9) # 8-byte Folded Spill
; GENDYN-NEXT:    or %s18, 0, %s0
; GENDYN-NEXT:    lea %s0, x@tls_gd_lo(-24)
; GENDYN-NEXT:    and %s0, %s0, (32)0
; GENDYN-NEXT:    sic %s10
; GENDYN-NEXT:    lea.sl %s0, x@tls_gd_hi(%s10, %s0)
; GENDYN-NEXT:    lea %s12, __tls_get_addr@plt_lo(8)
; GENDYN-NEXT:    and %s12, %s12, (32)0
; GENDYN-NEXT:    lea.sl %s12, __tls_get_addr@plt_hi(%s10, %s12)
; GENDYN-NEXT:    bsic %s10, (, %s12)
; GENDYN-NEXT:    stl %s18, (,%s0)
; GENDYN-NEXT:    ld %s18, 48(,%s9) # 8-byte Folded Reload
; GENDYN-NEXT:    or %s11, 0, %s9
; GENDYN-NEXT:    ld %s16, 32(,%s11)
; GENDYN-NEXT:    ld %s15, 24(,%s11)
; GENDYN-NEXT:    ld %s10, 8(,%s11)
; GENDYN-NEXT:    ld %s9, (,%s11)
; GENDYN-NEXT:    b.l (,%lr)
;
; GENDYNPIC-LABEL: set_global:
; GENDYNPIC:       # %bb.0: # %entry
; GENDYNPIC-NEXT:    st %s9, (,%s11)
; GENDYNPIC-NEXT:    st %s10, 8(,%s11)
; GENDYNPIC-NEXT:    st %s15, 24(,%s11)
; GENDYNPIC-NEXT:    st %s16, 32(,%s11)
; GENDYNPIC-NEXT:    or %s9, 0, %s11
; GENDYNPIC-NEXT:    lea %s13, -240
; GENDYNPIC-NEXT:    and %s13, %s13, (32)0
; GENDYNPIC-NEXT:    lea.sl %s11, -1(%s11, %s13)
; GENDYNPIC-NEXT:    brge.l %s11, %s8, .LBB2_2
; GENDYNPIC-NEXT:  # %bb.1: # %entry
; GENDYNPIC-NEXT:    ld %s61, 24(,%s14)
; GENDYNPIC-NEXT:    or %s62, 0, %s0
; GENDYNPIC-NEXT:    lea %s63, 315
; GENDYNPIC-NEXT:    shm.l %s63, (%s61)
; GENDYNPIC-NEXT:    shm.l %s8, 8(%s61)
; GENDYNPIC-NEXT:    shm.l %s11, 16(%s61)
; GENDYNPIC-NEXT:    monc
; GENDYNPIC-NEXT:    or %s0, 0, %s62
; GENDYNPIC-NEXT:  .LBB2_2: # %entry
; GENDYNPIC-NEXT:    st %s18, 48(,%s9) # 8-byte Folded Spill
; GENDYNPIC-NEXT:    or %s18, 0, %s0
; GENDYNPIC-NEXT:    lea %s15, _GLOBAL_OFFSET_TABLE_@pc_lo(-24)
; GENDYNPIC-NEXT:    and %s15, %s15, (32)0
; GENDYNPIC-NEXT:    sic %s16
; GENDYNPIC-NEXT:    lea.sl %s15, _GLOBAL_OFFSET_TABLE_@pc_hi(%s16, %s15)
; GENDYNPIC-NEXT:    lea %s0, x@tls_gd_lo(-24)
; GENDYNPIC-NEXT:    and %s0, %s0, (32)0
; GENDYNPIC-NEXT:    sic %s10
; GENDYNPIC-NEXT:    lea.sl %s0, x@tls_gd_hi(%s10, %s0)
; GENDYNPIC-NEXT:    lea %s12, __tls_get_addr@plt_lo(8)
; GENDYNPIC-NEXT:    and %s12, %s12, (32)0
; GENDYNPIC-NEXT:    lea.sl %s12, __tls_get_addr@plt_hi(%s10, %s12)
; GENDYNPIC-NEXT:    bsic %s10, (, %s12)
; GENDYNPIC-NEXT:    stl %s18, (,%s0)
; GENDYNPIC-NEXT:    ld %s18, 48(,%s9) # 8-byte Folded Reload
; GENDYNPIC-NEXT:    or %s11, 0, %s9
; GENDYNPIC-NEXT:    ld %s16, 32(,%s11)
; GENDYNPIC-NEXT:    ld %s15, 24(,%s11)
; GENDYNPIC-NEXT:    ld %s10, 8(,%s11)
; GENDYNPIC-NEXT:    ld %s9, (,%s11)
; GENDYNPIC-NEXT:    b.l (,%lr)
; LOCAL-LABEL: set_global:
; LOCAL:       .LBB{{[0-9]+}}_2:
; LOCAL-NEXT:  lea %s34, x@tpoff_lo
; LOCAL-NEXT:  and %s34, %s34, (32)0
; LOCAL-NEXT:  lea.sl %s34, x@tpoff_hi(%s34)
; LOCAL-NEXT:  adds.l %s34, %s14, %s34
; LOCAL-NEXT:  stl %s0, (,%s34)
; LOCAL-NEXT:  or %s11, 0, %s9
entry:
  store i32 %v, i32* @x, align 4
  ret void
}

; Function Attrs: norecurse nounwind
define void @set_local(i32 %v) {
; GENDYN-LABEL: set_local:
; GENDYN:       # %bb.0: # %entry
; GENDYN-NEXT:    st %s9, (,%s11)
; GENDYN-NEXT:    st %s10, 8(,%s11)
; GENDYN-NEXT:    st %s15, 24(,%s11)
; GENDYN-NEXT:    st %s16, 32(,%s11)
; GENDYN-NEXT:    or %s9, 0, %s11
; GENDYN-NEXT:    lea %s13, -240
; GENDYN-NEXT:    and %s13, %s13, (32)0
; GENDYN-NEXT:    lea.sl %s11, -1(%s11, %s13)
; GENDYN-NEXT:    brge.l %s11, %s8, .LBB3_2
; GENDYN-NEXT:  # %bb.1: # %entry
; GENDYN-NEXT:    ld %s61, 24(,%s14)
; GENDYN-NEXT:    or %s62, 0, %s0
; GENDYN-NEXT:    lea %s63, 315
; GENDYN-NEXT:    shm.l %s63, (%s61)
; GENDYN-NEXT:    shm.l %s8, 8(%s61)
; GENDYN-NEXT:    shm.l %s11, 16(%s61)
; GENDYN-NEXT:    monc
; GENDYN-NEXT:    or %s0, 0, %s62
; GENDYN-NEXT:  .LBB3_2: # %entry
; GENDYN-NEXT:    st %s18, 48(,%s9) # 8-byte Folded Spill
; GENDYN-NEXT:    or %s18, 0, %s0
; GENDYN-NEXT:    lea %s0, y@tls_gd_lo(-24)
; GENDYN-NEXT:    and %s0, %s0, (32)0
; GENDYN-NEXT:    sic %s10
; GENDYN-NEXT:    lea.sl %s0, y@tls_gd_hi(%s10, %s0)
; GENDYN-NEXT:    lea %s12, __tls_get_addr@plt_lo(8)
; GENDYN-NEXT:    and %s12, %s12, (32)0
; GENDYN-NEXT:    lea.sl %s12, __tls_get_addr@plt_hi(%s10, %s12)
; GENDYN-NEXT:    bsic %s10, (, %s12)
; GENDYN-NEXT:    stl %s18, (,%s0)
; GENDYN-NEXT:    ld %s18, 48(,%s9) # 8-byte Folded Reload
; GENDYN-NEXT:    or %s11, 0, %s9
; GENDYN-NEXT:    ld %s16, 32(,%s11)
; GENDYN-NEXT:    ld %s15, 24(,%s11)
; GENDYN-NEXT:    ld %s10, 8(,%s11)
; GENDYN-NEXT:    ld %s9, (,%s11)
; GENDYN-NEXT:    b.l (,%lr)
;
; GENDYNPIC-LABEL: set_local:
; GENDYNPIC:       # %bb.0: # %entry
; GENDYNPIC-NEXT:    st %s9, (,%s11)
; GENDYNPIC-NEXT:    st %s10, 8(,%s11)
; GENDYNPIC-NEXT:    st %s15, 24(,%s11)
; GENDYNPIC-NEXT:    st %s16, 32(,%s11)
; GENDYNPIC-NEXT:    or %s9, 0, %s11
; GENDYNPIC-NEXT:    lea %s13, -240
; GENDYNPIC-NEXT:    and %s13, %s13, (32)0
; GENDYNPIC-NEXT:    lea.sl %s11, -1(%s11, %s13)
; GENDYNPIC-NEXT:    brge.l %s11, %s8, .LBB3_2
; GENDYNPIC-NEXT:  # %bb.1: # %entry
; GENDYNPIC-NEXT:    ld %s61, 24(,%s14)
; GENDYNPIC-NEXT:    or %s62, 0, %s0
; GENDYNPIC-NEXT:    lea %s63, 315
; GENDYNPIC-NEXT:    shm.l %s63, (%s61)
; GENDYNPIC-NEXT:    shm.l %s8, 8(%s61)
; GENDYNPIC-NEXT:    shm.l %s11, 16(%s61)
; GENDYNPIC-NEXT:    monc
; GENDYNPIC-NEXT:    or %s0, 0, %s62
; GENDYNPIC-NEXT:  .LBB3_2: # %entry
; GENDYNPIC-NEXT:    st %s18, 48(,%s9) # 8-byte Folded Spill
; GENDYNPIC-NEXT:    or %s18, 0, %s0
; GENDYNPIC-NEXT:    lea %s15, _GLOBAL_OFFSET_TABLE_@pc_lo(-24)
; GENDYNPIC-NEXT:    and %s15, %s15, (32)0
; GENDYNPIC-NEXT:    sic %s16
; GENDYNPIC-NEXT:    lea.sl %s15, _GLOBAL_OFFSET_TABLE_@pc_hi(%s16, %s15)
; GENDYNPIC-NEXT:    lea %s0, y@tls_gd_lo(-24)
; GENDYNPIC-NEXT:    and %s0, %s0, (32)0
; GENDYNPIC-NEXT:    sic %s10
; GENDYNPIC-NEXT:    lea.sl %s0, y@tls_gd_hi(%s10, %s0)
; GENDYNPIC-NEXT:    lea %s12, __tls_get_addr@plt_lo(8)
; GENDYNPIC-NEXT:    and %s12, %s12, (32)0
; GENDYNPIC-NEXT:    lea.sl %s12, __tls_get_addr@plt_hi(%s10, %s12)
; GENDYNPIC-NEXT:    bsic %s10, (, %s12)
; GENDYNPIC-NEXT:    stl %s18, (,%s0)
; GENDYNPIC-NEXT:    ld %s18, 48(,%s9) # 8-byte Folded Reload
; GENDYNPIC-NEXT:    or %s11, 0, %s9
; GENDYNPIC-NEXT:    ld %s16, 32(,%s11)
; GENDYNPIC-NEXT:    ld %s15, 24(,%s11)
; GENDYNPIC-NEXT:    ld %s10, 8(,%s11)
; GENDYNPIC-NEXT:    ld %s9, (,%s11)
; GENDYNPIC-NEXT:    b.l (,%lr)
; LOCAL-LABEL: set_local:
; LOCAL:       .LBB{{[0-9]+}}_2:
; LOCAL-NEXT:  lea %s34, y@tpoff_lo
; LOCAL-NEXT:  and %s34, %s34, (32)0
; LOCAL-NEXT:  lea.sl %s34, y@tpoff_hi(%s34)
; LOCAL-NEXT:  adds.l %s34, %s14, %s34
; LOCAL-NEXT:  stl %s0, (,%s34)
; LOCAL-NEXT:  or %s11, 0, %s9
entry:
  store i32 %v, i32* @y, align 4
  ret void
}
