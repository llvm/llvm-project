; RUN: llc < %s -emulated-tls -mtriple=aarch64-linux-android -relocation-model=pic \
; RUN:     | FileCheck -check-prefix=ARM_64 %s
; RUN: llc < %s -emulated-tls -mtriple=aarch64-linux-android -relocation-model=pic -O3 \
; RUN:     | FileCheck -check-prefix=ARM_64 %s
; RUN: llc < %s -emulated-tls -mtriple=aarch64-linux-android -O3 \
; RUN:     | FileCheck -check-prefix=ARM_64 %s

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

; ARM_64-LABEL:  get_external_x:
; ARM_64:      __emutls_v.external_x
; ARM_64:      __emutls_get_address
; ARM_64-LABEL:  get_external_y:
; ARM_64:      __emutls_v.external_y
; ARM_64:      __emutls_get_address
; ARM_64-LABEL:  get_internal_y:
; ARM_64:      __emutls_v.internal_y
; ARM_64:      __emutls_get_address
; ARM_64-NOT:   __emutls_t.external_x
; ARM_64-NOT:   __emutls_v.external_x:
; ARM_64:        .data{{$}}
; ARM_64:        .globl __emutls_v.external_y
; ARM_64:        .p2align 3
; ARM_64-LABEL:  __emutls_v.external_y:
; ARM_64-NEXT:   .xword 1
; ARM_64-NEXT:   .xword 2
; ARM_64-NEXT:   .xword 0
; ARM_64-NEXT:   .xword __emutls_t.external_y
; ARM_64-NOT:    __emutls_v.external_x:
; ARM_64:        .section .rodata,
; ARM_64-LABEL:  __emutls_t.external_y:
; ARM_64-NEXT:   .byte 7
; ARM_64:        .data{{$}}
; ARM_64-NOT:    .globl __emutls_v
; ARM_64:        .p2align 3
; ARM_64-LABEL:  __emutls_v.internal_y:
; ARM_64-NEXT:   .xword 8
; ARM_64-NEXT:   .xword 16
; ARM_64-NEXT:   .xword 0
; ARM_64-NEXT:   .xword __emutls_t.internal_y
; ARM_64:        .section .rodata,
; ARM_64-LABEL:  __emutls_t.internal_y:
; ARM_64-NEXT:   .xword 9
