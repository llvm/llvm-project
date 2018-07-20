; RUN: llc < %s -emulated-tls -mtriple=i686-linux-android -relocation-model=pic | FileCheck  %s
; RUN: llc < %s -emulated-tls -mtriple=x86_64-linux-android -relocation-model=pic | FileCheck -check-prefix=X64 %s

; RUN: llc < %s -mtriple=i686-linux-android -relocation-model=pic | FileCheck  %s
; RUN: llc < %s -mtriple=x86_64-linux-android -relocation-model=pic | FileCheck -check-prefix=X64 %s

; Make sure that TLS symboles are emitted in expected order.

@external_x = external thread_local global i32
@external_y = thread_local global i32 7
@internal_y = internal thread_local global i32 9

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

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; 32-bit mode
; CHECK-LABEL: get_external_x:
; CHECK:  __emutls_v.external_x
; CHECK:  __emutls_get_address

; CHECK-LABEL: get_external_y:
; CHECK:  __emutls_v.external_y
; CHECK:  __emutls_get_address

; CHECK-LABEL: get_internal_y:
; CHECK:  __emutls_v.internal_y
; CHECK:  __emutls_get_address

; CHECK-NOT: __emutls_v.external_x:

; CHECK:       .p2align 2
; CHECK-LABEL: __emutls_v.external_y:
; CHECK-NEXT:  .long 4
; CHECK-NEXT:  .long 4
; CHECK-NEXT:  .long 0
; CHECK-NEXT:  .long __emutls_t.external_y
; CHECK-LABEL: __emutls_t.external_y:
; CHECK-NEXT:  .long 7

; CHECK:       .p2align 2
; CHECK-LABEL: __emutls_v.internal_y:
; CHECK-NEXT:  .long 4
; CHECK-NEXT:  .long 4
; CHECK-NEXT:  .long 0
; CHECK-NEXT:  .long __emutls_t.internal_y
; CHECK-LABEL: __emutls_t.internal_y:
; CHECK-NEXT:  .long 9

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; 64-bit mode
; X64-LABEL: get_external_x:
; X64:  __emutls_v.external_x
; X64:  __emutls_get_address

; X64-LABEL: get_external_y:
; X64:  __emutls_v.external_y
; X64:  __emutls_get_address

; X64-LABEL: get_internal_y:
; X64:  __emutls_v.internal_y
; X64:  __emutls_get_address

; X64-NOT: __emutls_v.external_x:

; X64:       .p2align 3
; X64-LABEL: __emutls_v.external_y:
; X64-NEXT:  .quad 4
; X64-NEXT:  .quad 4
; X64-NEXT:  .quad 0
; X64-NEXT:  .quad __emutls_t.external_y
; X64-LABEL: __emutls_t.external_y:
; X64-NEXT:  .long 7

; X64:       .p2align 3
; X64-LABEL: __emutls_v.internal_y:
; X64-NEXT:  .quad 4
; X64-NEXT:  .quad 4
; X64-NEXT:  .quad 0
; X64-NEXT:  .quad __emutls_t.internal_y
; X64-LABEL: __emutls_t.internal_y:
; X64-NEXT:  .long 9
