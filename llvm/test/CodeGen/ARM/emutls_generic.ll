; RUN: llc < %s -emulated-tls -mtriple=arm-linux-android -relocation-model=pic \
; RUN:     | FileCheck -check-prefix=ARM_32 %s
; RUN: llc < %s -emulated-tls -mtriple=arm-linux-androidabi -relocation-model=pic \
; RUN:     | FileCheck -check-prefix=ARM_32 %s
; RUN: llc < %s -emulated-tls -mtriple=arm-linux-androidabi -relocation-model=pic -O3 \
; RUN:     | FileCheck -check-prefix=ARM_32 %s
; RUN: llc < %s -emulated-tls -mtriple=arm-linux-androidabi -O3 \
; RUN:     | FileCheck -check-prefix=ARM_32 %s

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

; ARM_32-LABEL:  get_external_x:
; ARM_32:        bl __emutls_get_address
; ARM_32:        .long __emutls_v.external_x
; ARM_32-LABEL:  get_external_y:
; ARM_32:        bl __emutls_get_address
; ARM_32:        .long __emutls_v.external_y
; ARM_32-LABEL:  get_internal_y:
; ARM_32:      bl __emutls_get_address
; ARM_32:      .long __emutls_v.internal_y
; ARM_32-NOT:   __emutls_t.external_x
; ARM_32-NOT:   __emutls_v.external_x:
; ARM_32:        .section .data.rel
; ARM_32:        .align 2
; ARM_32-LABEL:  __emutls_v.external_y:
; ARM_32-NEXT:   .long 1
; ARM_32-NEXT:   .long 2
; ARM_32-NEXT:   .long 0
; ARM_32-NEXT:   .long __emutls_t.external_y
; ARM_32:        .section .rodata,
; ARM_32-LABEL:  __emutls_t.external_y:
; ARM_32-NEXT:   .byte 7
; ARM_32:        .section .data.rel
; ARM_32:        .align 2
; ARM_32-LABEL:  __emutls_v.internal_y:
; ARM_32-NEXT:   .long 8
; ARM_32-NEXT:   .long 16
; ARM_32-NEXT:   .long 0
; ARM_32-NEXT:   .long __emutls_t.internal_y
; ARM_32-LABEL:  __emutls_t.internal_y:
; ARM_32-NEXT:   .long 9
; ARM_32-NEXT:   .long 0
