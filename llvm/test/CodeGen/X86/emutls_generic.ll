; RUN: llc < %s -emulated-tls -mtriple=i686-linux-android -relocation-model=pic \
; RUN:     | FileCheck -check-prefix=X86_32 %s
; RUN: llc < %s -emulated-tls -mtriple=i686-linux-android -relocation-model=pic \
; RUN:     | FileCheck -check-prefix=X86_32 %s
; RUN: llc < %s -emulated-tls -mtriple=x86_64-linux-android -relocation-model=pic \
; RUN:     | FileCheck -check-prefix=X86_64 %s
; RUN: llc < %s -emulated-tls -mtriple=i386-linux-gnu -relocation-model=pic \
; RUN:     | FileCheck %s

; RUN: llc < %s -mtriple=i686-linux-android -relocation-model=pic \
; RUN:     | FileCheck -check-prefix=X86_32 %s
; RUN: llc < %s -mtriple=i686-linux-android -relocation-model=pic \
; RUN:     | FileCheck -check-prefix=X86_32 %s
; RUN: llc < %s -mtriple=x86_64-linux-android -relocation-model=pic \
; RUN:     | FileCheck -check-prefix=X86_64 %s
; RUN: llc < %s -mtriple=i386-linux-gnu -relocation-model=pic \
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

; X86_32-LABEL:  get_external_x:
; X86_32:        movl __emutls_v.external_x@GOT(%ebx)
; X86_32:        calll __emutls_get_address
; X86_32-LABEL:  get_external_y:
; X86_32:        movl __emutls_v.external_y@GOT(%ebx)
; X86_32:        calll __emutls_get_address
; X86_32-LABEL:  get_internal_y:
; X86_32:        leal __emutls_v.internal_y@GOTOFF(%ebx)
; X86_32:        calll __emutls_get_address
; X86_32-NOT:    __emutls_t.external_x
; X86_32-NOT:    __emutls_v.external_x:
; X86_32:        .data{{$}}
; X86_32:        .globl __emutls_v.external_y
; X86_32:        .p2align 2
; X86_32-LABEL:  __emutls_v.external_y:
; X86_32-NEXT:   .long 1
; X86_32-NEXT:   .long 2
; X86_32-NEXT:   .long 0
; X86_32-NEXT:   .long __emutls_t.external_y
; X86_32:        .section .rodata,
; X86_32-LABEL:  __emutls_t.external_y:
; X86_32-NEXT:   .byte 7
; X86_32:        .data{{$}}
; X86_32-NOT:    .globl
; X86_32:        .p2align 2
; X86_32-LABEL:  __emutls_v.internal_y:
; X86_32-NEXT:   .long 8
; X86_32-NEXT:   .long 16
; X86_32-NEXT:   .long 0
; X86_32-NEXT:   .long __emutls_t.internal_y
; X86_32-LABEL:  __emutls_t.internal_y:
; X86_32-NEXT:   .quad 9
; X86_64-LABEL:  get_external_x:
; X86_64:        __emutls_v.external_x@GOTPCREL(%rip)
; X86_64:        __emutls_get_address
; X86_64-LABEL:  get_external_y:
; X86_64:        __emutls_v.external_y@GOTPCREL(%rip)
; X86_64:        __emutls_get_address
; X86_64-LABEL:  get_internal_y:
; X86_64:        __emutls_v.internal_y(%rip)
; X86_64:        __emutls_get_address
; X86_64-NOT:    __emutls_t.external_x
; X86_64-NOT:    __emutls_v.external_x:
; X86_64:        .globl __emutls_v.external_y
; X86_64:        .p2align 3
; X86_64-LABEL:  __emutls_v.external_y:
; X86_64-NEXT:   .quad 1
; X86_64-NEXT:   .quad 2
; X86_64-NEXT:   .quad 0
; X86_64-NEXT:   .quad __emutls_t.external_y
; X86_64-NOT:    __emutls_v.external_x:
; X86_64:        .section .rodata,
; X86_64-LABEL:  __emutls_t.external_y:
; X86_64-NEXT:   .byte 7
; X86_64:        .data{{$}}
; X86_64-NOT:    .globl
; X86_64:        .p2align 3
; X86_64-LABEL:  __emutls_v.internal_y:
; X86_64-NEXT:   .quad 8
; X86_64-NEXT:   .quad 16
; X86_64-NEXT:   .quad 0
; X86_64-NEXT:   .quad __emutls_t.internal_y
; X86_64:        .section .rodata,
; X86_64-LABEL:  __emutls_t.internal_y:
; X86_64-NEXT:   .quad 9
