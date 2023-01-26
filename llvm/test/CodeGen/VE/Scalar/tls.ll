; FIXME: even under non-pic mode, llvm needs to generate pic code since nld
;        doesn't work with non-pic code.  Thefore, we test pic codes for
;        both cases here.
;      llc -mtriple ve < %s | FileCheck %s -check-prefix=LOCAL
; RUN: llc -mtriple ve < %s | FileCheck %s -check-prefix=GENDYN
; RUN: llc -mtriple ve -relocation-model=pic < %s | FileCheck %s -check-prefix=GENDYNPIC

@x = external thread_local global i32, align 4
@y = internal thread_local global i32 0, align 4

; Function Attrs: norecurse nounwind readnone
define nonnull ptr @get_global() {
; GENDYN-LABEL: get_global:
; GENDYN:       .LBB{{[0-9]+}}_2:
; GENDYN-NEXT:    lea %s0, x@tls_gd_lo(-24)
; GENDYN-NEXT:    and %s0, %s0, (32)0
; GENDYN-NEXT:    sic %s10
; GENDYN-NEXT:    lea.sl %s0, x@tls_gd_hi(%s10, %s0)
; GENDYN-NEXT:    lea %s12, __tls_get_addr@plt_lo(8)
; GENDYN-NEXT:    and %s12, %s12, (32)0
; GENDYN-NEXT:    lea.sl %s12, __tls_get_addr@plt_hi(%s10, %s12)
; GENDYN-NEXT:    bsic %s10, (, %s12)
; GENDYN-NEXT:    or %s11, 0, %s9
;
; GENDYNPIC-LABEL: get_global:
; GENDYNPIC:       .LBB{{[0-9]+}}_2:
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
;
; LOCAL-LABEL: get_global:
; LOCAL:       .LBB{{[0-9]+}}_2:
; LOCAL-NEXT:  lea %s34, x@tpoff_lo
; LOCAL-NEXT:  and %s34, %s34, (32)0
; LOCAL-NEXT:  lea.sl %s34, x@tpoff_hi(%s34)
; LOCAL-NEXT:  adds.l %s0, %s14, %s34
; LOCAL-NEXT:  or %s11, 0, %s9
entry:
  ret ptr @x
}

; Function Attrs: norecurse nounwind readnone
define nonnull ptr @get_local() {
; GENDYN-LABEL: get_local:
; GENDYN:       .LBB{{[0-9]+}}_2:
; GENDYN-NEXT:    lea %s0, y@tls_gd_lo(-24)
; GENDYN-NEXT:    and %s0, %s0, (32)0
; GENDYN-NEXT:    sic %s10
; GENDYN-NEXT:    lea.sl %s0, y@tls_gd_hi(%s10, %s0)
; GENDYN-NEXT:    lea %s12, __tls_get_addr@plt_lo(8)
; GENDYN-NEXT:    and %s12, %s12, (32)0
; GENDYN-NEXT:    lea.sl %s12, __tls_get_addr@plt_hi(%s10, %s12)
; GENDYN-NEXT:    bsic %s10, (, %s12)
; GENDYN-NEXT:    or %s11, 0, %s9
;
; GENDYNPIC-LABEL: get_local:
; GENDYNPIC:       .LBB{{[0-9]+}}_2:
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
;
; LOCAL-LABEL: get_local:
; LOCAL:       .LBB{{[0-9]+}}_2:
; LOCAL-NEXT:  lea %s34, y@tpoff_lo
; LOCAL-NEXT:  and %s34, %s34, (32)0
; LOCAL-NEXT:  lea.sl %s34, y@tpoff_hi(%s34)
; LOCAL-NEXT:  adds.l %s0, %s14, %s34
; LOCAL-NEXT:  or %s11, 0, %s9
entry:
  ret ptr @y
}

; Function Attrs: norecurse nounwind
define void @set_global(i32 %v) {
; GENDYN-LABEL: set_global:
; GENDYN:       .LBB{{[0-9]+}}_2:
; GENDYN-NEXT:    st %s18, 288(, %s11) # 8-byte Folded Spill
; GENDYN-NEXT:    or %s18, 0, %s0
; GENDYN-NEXT:    lea %s0, x@tls_gd_lo(-24)
; GENDYN-NEXT:    and %s0, %s0, (32)0
; GENDYN-NEXT:    sic %s10
; GENDYN-NEXT:    lea.sl %s0, x@tls_gd_hi(%s10, %s0)
; GENDYN-NEXT:    lea %s12, __tls_get_addr@plt_lo(8)
; GENDYN-NEXT:    and %s12, %s12, (32)0
; GENDYN-NEXT:    lea.sl %s12, __tls_get_addr@plt_hi(%s10, %s12)
; GENDYN-NEXT:    bsic %s10, (, %s12)
; GENDYN-NEXT:    stl %s18, (, %s0)
; GENDYN-NEXT:    ld %s18, 288(, %s11) # 8-byte Folded Reload
; GENDYN-NEXT:    or %s11, 0, %s9
;
; GENDYNPIC-LABEL: set_global:
; GENDYNPIC:       .LBB{{[0-9]+}}_2:
; GENDYNPIC-NEXT:    st %s18, 288(, %s11) # 8-byte Folded Spill
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
; GENDYNPIC-NEXT:    stl %s18, (, %s0)
; GENDYNPIC-NEXT:    ld %s18, 288(, %s11) # 8-byte Folded Reload
; GENDYNPIC-NEXT:    or %s11, 0, %s9
;
; LOCAL-LABEL: set_global:
; LOCAL:       .LBB{{[0-9]+}}_2:
; LOCAL-NEXT:  lea %s34, x@tpoff_lo
; LOCAL-NEXT:  and %s34, %s34, (32)0
; LOCAL-NEXT:  lea.sl %s34, x@tpoff_hi(%s34)
; LOCAL-NEXT:  adds.l %s34, %s14, %s34
; LOCAL-NEXT:  stl %s0, (, %s34)
; LOCAL-NEXT:  or %s11, 0, %s9
entry:
  store i32 %v, ptr @x, align 4
  ret void
}

; Function Attrs: norecurse nounwind
define void @set_local(i32 %v) {
; GENDYN-LABEL: set_local:
; GENDYN:       .LBB{{[0-9]+}}_2:
; GENDYN-NEXT:    st %s18, 288(, %s11) # 8-byte Folded Spill
; GENDYN-NEXT:    or %s18, 0, %s0
; GENDYN-NEXT:    lea %s0, y@tls_gd_lo(-24)
; GENDYN-NEXT:    and %s0, %s0, (32)0
; GENDYN-NEXT:    sic %s10
; GENDYN-NEXT:    lea.sl %s0, y@tls_gd_hi(%s10, %s0)
; GENDYN-NEXT:    lea %s12, __tls_get_addr@plt_lo(8)
; GENDYN-NEXT:    and %s12, %s12, (32)0
; GENDYN-NEXT:    lea.sl %s12, __tls_get_addr@plt_hi(%s10, %s12)
; GENDYN-NEXT:    bsic %s10, (, %s12)
; GENDYN-NEXT:    stl %s18, (, %s0)
; GENDYN-NEXT:    ld %s18, 288(, %s11) # 8-byte Folded Reload
; GENDYN-NEXT:    or %s11, 0, %s9
;
; GENDYNPIC-LABEL: set_local:
; GENDYNPIC:       .LBB{{[0-9]+}}_2:
; GENDYNPIC-NEXT:    st %s18, 288(, %s11) # 8-byte Folded Spill
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
; GENDYNPIC-NEXT:    stl %s18, (, %s0)
; GENDYNPIC-NEXT:    ld %s18, 288(, %s11) # 8-byte Folded Reload
; GENDYNPIC-NEXT:    or %s11, 0, %s9
;
; LOCAL-LABEL: set_local:
; LOCAL:       .LBB{{[0-9]+}}_2:
; LOCAL-NEXT:  lea %s34, y@tpoff_lo
; LOCAL-NEXT:  and %s34, %s34, (32)0
; LOCAL-NEXT:  lea.sl %s34, y@tpoff_hi(%s34)
; LOCAL-NEXT:  adds.l %s34, %s14, %s34
; LOCAL-NEXT:  stl %s0, (, %s34)
; LOCAL-NEXT:  or %s11, 0, %s9
entry:
  store i32 %v, ptr @y, align 4
  ret void
}
