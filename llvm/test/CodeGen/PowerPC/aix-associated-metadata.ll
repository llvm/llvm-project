; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s
; RUN: llc -data-sections=false -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck -check-prefix=NODATA %s

@a = global i32 1
@b = global i32 2
@c = global i32 3, section "custom_section_c"
@d = global i32 4, !associated !0
@e = constant i32 5, !associated !1, !associated !2
@f = global i32 6, section "custom_section_f", !associated !1


!0 = !{ptr @a}
!1 = !{ptr @b}
!2 = !{ptr @c}

; CHECK: .csect d[RW]
; CHECK: .ref a[RW]

; CHECK: .csect e[RO]
; CHECK: .ref b[RW]
; CHECK: .ref c

; CHECK: .csect custom_section_f[RW]
; CHECK: .ref b[RW]

; NODATA:      .csect .data[RW]
; NODATA-NOT:  .csect
; NODATA:      .globl a
; NODATA-NOT:  .csect
; NODATA:      .globl b
; NODATA:      .csect custom_section_c[RW]
; NODATA:      .globl c
; NODATA:      .csect .data[RW]
; NODATA:      .ref a
; NODATA:      .globl d
; NODATA:      .csect .rodata[RO]
; NODATA:      .ref b
; NODATA:      .ref c
; NODATA:      .globl e
; NODATA:      .csect custom_section_f[RW]
; NODATA:      .ref b
; NODATA:      .globl f
