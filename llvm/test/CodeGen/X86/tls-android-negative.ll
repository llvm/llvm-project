; RUN: llc < %s -emulated-tls -march=x86 -mtriple=x86_64-linux-android -relocation-model=pic | FileCheck  %s
; RUN: llc < %s -emulated-tls -march=x86-64 -mtriple=x86_64-linux-android -relocation-model=pic | FileCheck  %s

; Make sure that some symboles are not emitted in emulated TLS model.

@external_x = external thread_local global i32
@external_y = thread_local global i32 7
@internal_y = internal thread_local global i32 9
@internal_y0 = internal thread_local global i32 0

define i32* @get_external_x() {
entry:
  ret i32* @external_x
}

define i32* @get_external_y() {
entry:
  ret i32* @external_y
}

define i32* @get_internal_y() {
entry:
  ret i32* @internal_y
}

define i32* @get_internal_y0() {
entry:
  ret i32* @internal_y0
}

; no direct access to emulated TLS variables.
; no definition of emulated TLS variables.
; no initializer for external TLS variables, __emutls_t.external_x
; no initializer for 0-initialized TLS variables, __emutls_t.internal_y0
; not global linkage for __emutls_t.external_y

; CHECK-NOT: external_x@TLS
; CHECK-NOT: external_y@TLS
; CHECK-NOT: internal_y@TLS
; CHECK-NOT: .size external_x
; CHECK-NOT: .size external_y
; CHECK-NOT: .size internal_y
; CHECK-NOT: .size internal_y0
; CHECK-NOT: __emutls_v.external_x:
; CHECK-NOT: __emutls_t.external_x:
; CHECK-NOT: __emutls_t.internal_y0:
; CHECK-NOT: global __emutls_t.external_y
; CHECK-NOT: global __emutls_v.internal_y
; CHECK-NOT: global __emutls_v.internal_y0

; CHECK:     __emutls_t.external_y

; CHECK-NOT: external_x@TLS
; CHECK-NOT: external_y@TLS
; CHECK-NOT: internal_y@TLS
; CHECK-NOT: .size external_x
; CHECK-NOT: .size external_y
; CHECK-NOT: .size internal_y
; CHECK-NOT: .size internal_y0
; CHECK-NOT: __emutls_v.external_x:
; CHECK-NOT: __emutls_t.external_x:
; CHECK-NOT: __emutls_t.internal_y0:
; CHECK-NOT: global __emutls_t.external_y
; CHECK-NOT: global __emutls_v.internal_y
; CHECK-NOT: global __emutls_v.internal_y0
