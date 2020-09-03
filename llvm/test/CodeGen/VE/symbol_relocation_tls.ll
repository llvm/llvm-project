; FIXME: Even under non-pic mode, llvm for ve needs to generate pic code since
;        nld doesn't work with non-pic code.  Thefore, we test only pic codes
;        for both cases here.
;      llc -filetype=obj -mtriple=ve -o - %s | llvm-objdump - -d -r \
;          | FileCheck %s -check-prefix=LOCAL
; RUN: llc -filetype=obj -mtriple=ve -o - %s | llvm-objdump - -d -r \
; RUN:     | FileCheck %s -check-prefix=GENDYN
; RUN: llc -filetype=obj -mtriple=ve -relocation-model=pic -o - %s \
; RUN:     |  llvm-objdump - -d -r | FileCheck %s -check-prefix=GENDYNPIC

@x = external thread_local global i32, align 4
@y = internal thread_local global i32 0, align 4

; Function Attrs: norecurse nounwind readnone
define nonnull i32* @get_global() {
; GENDYN:         lea %s0, (-24)
; GENDYN-NEXT:    R_VE_TLS_GD_LO32 x
; GENDYN-NEXT:    and %s0, %s0, (32)0
; GENDYN-NEXT:    sic %s10
; GENDYN-NEXT:    lea.sl %s0, (%s10, %s0)
; GENDYN-NEXT:    R_VE_TLS_GD_HI32 x
; GENDYN-NEXT:    lea %s12, (8)
; GENDYN-NEXT:    R_VE_PLT_LO32 __tls_get_addr
; GENDYN-NEXT:    and %s12, %s12, (32)0
; GENDYN-NEXT:    lea.sl %s12, (%s10, %s12)
; GENDYN-NEXT:    R_VE_PLT_HI32 __tls_get_addr
; GENDYN-NEXT:    bsic %s10, (, %s12)
; GENDYN-NEXT:    or %s11, 0, %s9
;
; GENDYNPIC:         lea %s15, (-24)
; GENDYNPIC-NEXT:    R_VE_PC_LO32 _GLOBAL_OFFSET_TABLE_
; GENDYNPIC-NEXT:    and %s15, %s15, (32)0
; GENDYNPIC-NEXT:    sic %s16
; GENDYNPIC-NEXT:    lea.sl %s15, (%s16, %s15)
; GENDYNPIC-NEXT:    R_VE_PC_HI32 _GLOBAL_OFFSET_TABLE_
; GENDYNPIC-NEXT:    lea %s0, (-24)
; GENDYNPIC-NEXT:    R_VE_TLS_GD_LO32 x
; GENDYNPIC-NEXT:    and %s0, %s0, (32)0
; GENDYNPIC-NEXT:    sic %s10
; GENDYNPIC-NEXT:    lea.sl %s0, (%s10, %s0)
; GENDYNPIC-NEXT:    R_VE_TLS_GD_HI32 x
; GENDYNPIC-NEXT:    lea %s12, (8)
; GENDYNPIC-NEXT:    R_VE_PLT_LO32 __tls_get_addr
; GENDYNPIC-NEXT:    and %s12, %s12, (32)0
; GENDYNPIC-NEXT:    lea.sl %s12, (%s10, %s12)
; GENDYNPIC-NEXT:    R_VE_PLT_HI32 __tls_get_addr
; GENDYNPIC-NEXT:    bsic %s10, (, %s12)
; GENDYNPIC-NEXT:    or %s11, 0, %s9
entry:
  ret i32* @x
}

; Function Attrs: norecurse nounwind readnone
define nonnull i32* @get_local() {
; GENDYN:         lea %s0, (-24)
; GENDYN-NEXT:    R_VE_TLS_GD_LO32 y
; GENDYN-NEXT:    and %s0, %s0, (32)0
; GENDYN-NEXT:    sic %s10
; GENDYN-NEXT:    lea.sl %s0, (%s10, %s0)
; GENDYN-NEXT:    R_VE_TLS_GD_HI32 y
; GENDYN-NEXT:    lea %s12, (8)
; GENDYN-NEXT:    R_VE_PLT_LO32 __tls_get_addr
; GENDYN-NEXT:    and %s12, %s12, (32)0
; GENDYN-NEXT:    lea.sl %s12, (%s10, %s12)
; GENDYN-NEXT:    R_VE_PLT_HI32 __tls_get_addr
; GENDYN-NEXT:    bsic %s10, (, %s12)
; GENDYN-NEXT:    or %s11, 0, %s9
;
; GENDYNPIC:         lea %s15, (-24)
; GENDYNPIC-NEXT:    R_VE_PC_LO32 _GLOBAL_OFFSET_TABLE_
; GENDYNPIC-NEXT:    and %s15, %s15, (32)0
; GENDYNPIC-NEXT:    sic %s16
; GENDYNPIC-NEXT:    lea.sl %s15, (%s16, %s15)
; GENDYNPIC-NEXT:    R_VE_PC_HI32 _GLOBAL_OFFSET_TABLE_
; GENDYNPIC-NEXT:    lea %s0, (-24)
; GENDYNPIC-NEXT:    R_VE_TLS_GD_LO32 y
; GENDYNPIC-NEXT:    and %s0, %s0, (32)0
; GENDYNPIC-NEXT:    sic %s10
; GENDYNPIC-NEXT:    lea.sl %s0, (%s10, %s0)
; GENDYNPIC-NEXT:    R_VE_TLS_GD_HI32 y
; GENDYNPIC-NEXT:    lea %s12, (8)
; GENDYNPIC-NEXT:    R_VE_PLT_LO32 __tls_get_addr
; GENDYNPIC-NEXT:    and %s12, %s12, (32)0
; GENDYNPIC-NEXT:    lea.sl %s12, (%s10, %s12)
; GENDYNPIC-NEXT:    R_VE_PLT_HI32 __tls_get_addr
; GENDYNPIC-NEXT:    bsic %s10, (, %s12)
; GENDYNPIC-NEXT:    or %s11, 0, %s9
entry:
  ret i32* @y
}

; Function Attrs: norecurse nounwind
define void @set_global(i32 %v) {
; GENDYN:         lea %s0, (-24)
; GENDYN-NEXT:    R_VE_TLS_GD_LO32 x
; GENDYN-NEXT:    and %s0, %s0, (32)0
; GENDYN-NEXT:    sic %s10
; GENDYN-NEXT:    lea.sl %s0, (%s10, %s0)
; GENDYN-NEXT:    R_VE_TLS_GD_HI32 x
; GENDYN-NEXT:    lea %s12, (8)
; GENDYN-NEXT:    R_VE_PLT_LO32 __tls_get_addr
; GENDYN-NEXT:    and %s12, %s12, (32)0
; GENDYN-NEXT:    lea.sl %s12, (%s10, %s12)
; GENDYN-NEXT:    R_VE_PLT_HI32 __tls_get_addr
; GENDYN-NEXT:    bsic %s10, (, %s12)
; GENDYN-NEXT:    stl %s18, (, %s0)
; GENDYN-NEXT:    ld %s18, 48(, %s9)
; GENDYN-NEXT:    or %s11, 0, %s9
;
; GENDYNPIC:         lea %s15, (-24)
; GENDYNPIC-NEXT:    R_VE_PC_LO32 _GLOBAL_OFFSET_TABLE_
; GENDYNPIC-NEXT:    and %s15, %s15, (32)0
; GENDYNPIC-NEXT:    sic %s16
; GENDYNPIC-NEXT:    lea.sl %s15, (%s16, %s15)
; GENDYNPIC-NEXT:    R_VE_PC_HI32 _GLOBAL_OFFSET_TABLE_
; GENDYNPIC-NEXT:    lea %s0, (-24)
; GENDYNPIC-NEXT:    R_VE_TLS_GD_LO32 x
; GENDYNPIC-NEXT:    and %s0, %s0, (32)0
; GENDYNPIC-NEXT:    sic %s10
; GENDYNPIC-NEXT:    lea.sl %s0, (%s10, %s0)
; GENDYNPIC-NEXT:    R_VE_TLS_GD_HI32 x
; GENDYNPIC-NEXT:    lea %s12, (8)
; GENDYNPIC-NEXT:    R_VE_PLT_LO32 __tls_get_addr
; GENDYNPIC-NEXT:    and %s12, %s12, (32)0
; GENDYNPIC-NEXT:    lea.sl %s12, (%s10, %s12)
; GENDYNPIC-NEXT:    R_VE_PLT_HI32 __tls_get_addr
; GENDYNPIC-NEXT:    bsic %s10, (, %s12)
; GENDYNPIC-NEXT:    stl %s18, (, %s0)
; GENDYNPIC-NEXT:    ld %s18, 48(, %s9)
; GENDYNPIC-NEXT:    or %s11, 0, %s9
entry:
  store i32 %v, i32* @x, align 4
  ret void
}

; Function Attrs: norecurse nounwind
define void @set_local(i32 %v) {
; GENDYN:         lea %s0, (-24)
; GENDYN-NEXT:    R_VE_TLS_GD_LO32 y
; GENDYN-NEXT:    and %s0, %s0, (32)0
; GENDYN-NEXT:    sic %s10
; GENDYN-NEXT:    lea.sl %s0, (%s10, %s0)
; GENDYN-NEXT:    R_VE_TLS_GD_HI32 y
; GENDYN-NEXT:    lea %s12, (8)
; GENDYN-NEXT:    R_VE_PLT_LO32 __tls_get_addr
; GENDYN-NEXT:    and %s12, %s12, (32)0
; GENDYN-NEXT:    lea.sl %s12, (%s10, %s12)
; GENDYN-NEXT:    R_VE_PLT_HI32 __tls_get_addr
; GENDYN-NEXT:    bsic %s10, (, %s12)
; GENDYN-NEXT:    stl %s18, (, %s0)
; GENDYN-NEXT:    ld %s18, 48(, %s9)
; GENDYN-NEXT:    or %s11, 0, %s9
;
; GENDYNPIC:         lea %s15, (-24)
; GENDYNPIC-NEXT:    R_VE_PC_LO32 _GLOBAL_OFFSET_TABLE_
; GENDYNPIC-NEXT:    and %s15, %s15, (32)0
; GENDYNPIC-NEXT:    sic %s16
; GENDYNPIC-NEXT:    lea.sl %s15, (%s16, %s15)
; GENDYNPIC-NEXT:    R_VE_PC_HI32 _GLOBAL_OFFSET_TABLE_
; GENDYNPIC-NEXT:    lea %s0, (-24)
; GENDYNPIC-NEXT:    R_VE_TLS_GD_LO32 y
; GENDYNPIC-NEXT:    and %s0, %s0, (32)0
; GENDYNPIC-NEXT:    sic %s10
; GENDYNPIC-NEXT:    lea.sl %s0, (%s10, %s0)
; GENDYNPIC-NEXT:    R_VE_TLS_GD_HI32 y
; GENDYNPIC-NEXT:    lea %s12, (8)
; GENDYNPIC-NEXT:    R_VE_PLT_LO32 __tls_get_addr
; GENDYNPIC-NEXT:    and %s12, %s12, (32)0
; GENDYNPIC-NEXT:    lea.sl %s12, (%s10, %s12)
; GENDYNPIC-NEXT:    R_VE_PLT_HI32 __tls_get_addr
; GENDYNPIC-NEXT:    bsic %s10, (, %s12)
; GENDYNPIC-NEXT:    stl %s18, (, %s0)
; GENDYNPIC-NEXT:    ld %s18, 48(, %s9)
; GENDYNPIC-NEXT:    or %s11, 0, %s9
entry:
  store i32 %v, i32* @y, align 4
  ret void
}
