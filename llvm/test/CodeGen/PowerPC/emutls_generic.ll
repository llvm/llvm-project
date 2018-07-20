; RUN: llc < %s -emulated-tls -mtriple=powerpc64-unknown-linux-gnu -relocation-model=pic \
; RUN:     | FileCheck %s
; RUN: llc < %s -emulated-tls -mtriple=powerpc-unknown-linux-gnu -relocation-model=pic \
; RUN:     | FileCheck %s

; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -relocation-model=pic \
; RUN:     | FileCheck -check-prefix=NoEMU %s
; RUN: llc < %s -mtriple=powerpc-unknown-linux-gnu -relocation-model=pic \
; RUN:     | FileCheck -check-prefix=NoEMU %s

; NoEMU-NOT: __emutls

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

; CHECK-LABEL: get_external_x:
; CHECK-NOT:   _tls_get_address
; CHECK:       __emutls_get_address
; CHECK-LABEL: get_external_y:
; CHECK:       __emutls_get_address
; CHECK-NOT:   _tls_get_address
; CHECK-LABEL: get_internal_y:
; CHECK-NOT:   __emutls_t.external_x:
; CHECK-NOT:   __emutls_v.external_x:
; CHECK-LABEL: __emutls_v.external_y:
; CHECK-LABEL: __emutls_t.external_y:
; CHECK:       __emutls_t.external_y
; CHECK-LABEL: __emutls_v.internal_y:
; CHECK-LABEL: __emutls_t.internal_y:
; CHECK:       __emutls_t.internal_y
