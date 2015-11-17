; RUN: llc < %s -emulated-tls -mtriple=mipsel-linux-android -relocation-model=pic \
; RUN:     | FileCheck -check-prefix=MIPS_32 %s
; RUN: llc < %s -emulated-tls -mtriple=mips64el-linux-android -relocation-model=pic \
; RUN:     | FileCheck -check-prefix=MIPS_64 %s

; Make sure that TLS symbols are emitted in expected order.

@external_x = external thread_local global i32, align 8
@external_y = thread_local global i8 7, align 2
@internal_y = internal thread_local global i64 9, align 16

define i32* @get_external_x() {
entry:
  ret i32* @external_x
}

define i8* @get_external_y() {
entry:
  ret i8* @external_y
}

define i64* @get_internal_y() {
entry:
  ret i64* @internal_y
}

; MIPS_32-LABEL: get_external_y:
; MIPS_32-LABEL: get_internal_y:
; MIPS_32:     lw {{.+}}(__emutls_v.internal_y
; MIPS_32:     lw {{.+}}call16(__emutls_get_address
; MIPS_32-NOT:  __emutls_t.external_x
; MIPS_32-NOT:  __emutls_v.external_x:
; MIPS_32:       .section .data.rel
; MIPS_32:       .align 2
; MIPS_32-LABEL: __emutls_v.external_y:
; MIPS_32:       .section .rodata,
; MIPS_32-LABEL: __emutls_t.external_y:
; MIPS_32-NEXT:  .byte 7
; MIPS_32:       .section .data.rel
; MIPS_32:       .align 2
; MIPS_32-LABEL: __emutls_v.internal_y:
; MIPS_32-NEXT:  .4byte 8
; MIPS_32-NEXT:  .4byte 16
; MIPS_32-NEXT:  .4byte 0
; MIPS_32-NEXT:  .4byte __emutls_t.internal_y
; MIPS_32-LABEL: __emutls_t.internal_y:
; MIPS_32-NEXT:  .8byte 9

; MIPS_64-LABEL: get_external_x:
; MIPS_64-LABEL: get_external_y:
; MIPS_64-LABEL: get_internal_y:
; MIPS_64:     ld {{.+}}(__emutls_v.internal_y
; MIPS_64:     ld {{.+}}call16(__emutls_get_address
; MIPS_64-NOT:  __emutls_t.external_x
; MIPS_64-NOT:  __emutls_v.external_x:
; MIPS_64-LABEL: __emutls_v.external_y:
; MIPS_64-NOT:   __emutls_v.external_x:
; MIPS_64:       .section .rodata,
; MIPS_64-LABEL: __emutls_t.external_y:
; MIPS_64-NEXT:  .byte 7
; MIPS_64:       .section .data.rel
; MIPS_64:       .align 3
; MIPS_64-LABEL: __emutls_v.internal_y:
; MIPS_64-NEXT:  .8byte 8
; MIPS_64-NEXT:  .8byte 16
; MIPS_64-NEXT:  .8byte 0
; MIPS_64-NEXT:  .8byte __emutls_t.internal_y
; MIPS_64:       .section .rodata,
; MIPS_64-LABEL: __emutls_t.internal_y:
; MIPS_64-NEXT:  .8byte 9
