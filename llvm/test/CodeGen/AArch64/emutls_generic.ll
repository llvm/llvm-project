; RUN: llc < %s -emulated-tls -mtriple=aarch64-linux-android -relocation-model=pic \
; RUN:     | FileCheck -check-prefix=ARM_64 %s
; RUN: llc < %s -emulated-tls -mtriple=aarch64-linux-android -relocation-model=pic -O3 \
; RUN:     | FileCheck -check-prefix=ARM_64 %s
; RUN: llc < %s -emulated-tls -mtriple=aarch64-linux-android -O3 \
; RUN:     | FileCheck -check-prefix=ARM_64 %s
; RUN: llc < %s -emulated-tls -mtriple=aarch64-windows-gnu -O3 \
; RUN:     | FileCheck -check-prefix=ARM_64 %s
; RUN: llc < %s -emulated-tls -mtriple=aarch64-apple-darwin -O3 \
; RUN:     | FileCheck -check-prefix=DARWIN %s

; RUN: llc < %s -mtriple=aarch64-linux-android -relocation-model=pic \
; RUN:     | FileCheck -check-prefix=ARM_64 %s
; RUN: llc < %s -mtriple=aarch64-linux-android -relocation-model=pic -O3 \
; RUN:     | FileCheck -check-prefix=ARM_64 %s
; RUN: llc < %s -mtriple=aarch64-linux-android -O3 \
; RUN:     | FileCheck -check-prefix=ARM_64 %s
; aarch64-windows-gnu needs explicit -emulated-tls
; RUN: llc < %s -mtriple=aarch64-apple-darwin -O3 \
; RUN:     | FileCheck -check-prefix=NoEMU %s
; RUN: llc < %s -mtriple=aarch64-linux-android29 -O3 \
; RUN:     | FileCheck -check-prefix=NoEMU %s

; NoEMU-NOT: __emutls

; Make sure that TLS symbols are emitted in expected order.

@external_x = external thread_local global i32, align 8
@external_y = thread_local global i8 7, align 2
@internal_y = internal thread_local global i64 9, align 16

define ptr @get_external_x() {
entry:
  ret ptr @external_x
}

define ptr @get_external_y() {
entry:
  ret ptr @external_y
}

define ptr @get_internal_y() {
entry:
  ret ptr @internal_y
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
; ARM_64:        .section .r{{o?}}data,
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
; ARM_64:        .section .r{{o?}}data,
; ARM_64-LABEL:  __emutls_t.internal_y:
; ARM_64-NEXT:   .xword 9

; DARWIN-LABEL:  _get_external_x:
; DARWIN:        ___emutls_v.external_x
; DARWIN:        ___emutls_get_address
; DARWIN-LABEL:  _get_external_y:
; DARWIN:        ___emutls_v.external_y
; DARWIN:        ___emutls_get_address
; DARWIN-LABEL:  _get_internal_y:
; DARWIN:        ___emutls_v.internal_y
; DARWIN:        ___emutls_get_address
; DARWIN-NOT:    ___emutls_t.external_x
; DARWIN-NOT:    ___emutls_v.external_x:
; DARWIN:        .section __DATA,__data
; DARWIN:        .globl ___emutls_v.external_y
; DARWIN:        .p2align 3
; DARWIN-LABEL:  ___emutls_v.external_y:
; DARWIN-NEXT:   .quad 1
; DARWIN-NEXT:   .quad 2
; DARWIN-NEXT:   .quad 0
; DARWIN-NEXT:   .quad ___emutls_t.external_y
; DARWIN-NOT:    ___emutls_v.external_x:
; DARWIN:        .section __TEXT,__const
; DARWIN-LABEL:  ___emutls_t.external_y:
; DARWIN-NEXT:   .byte 7
; DARWIN:        .section __DATA,__data
; DARWIN-NOT:    .globl ___emutls_v
; DARWIN:        .p2align 3
; DARWIN-LABEL:  ___emutls_v.internal_y:
; DARWIN-NEXT:   .quad 8
; DARWIN-NEXT:   .quad 16
; DARWIN-NEXT:   .quad 0
; DARWIN-NEXT:   .quad ___emutls_t.internal_y
; DARWIN:        .section __TEXT,__const
; DARWIN-LABEL:  ___emutls_t.internal_y:
; DARWIN-NEXT:   .quad 9
