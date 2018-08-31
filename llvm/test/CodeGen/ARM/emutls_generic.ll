; RUN: llc < %s -emulated-tls -mtriple=arm-linux-android -relocation-model=pic \
; RUN:     | FileCheck -check-prefix=ARM_32 %s
; RUN: llc < %s -emulated-tls -mtriple=arm-linux-androidabi -relocation-model=pic \
; RUN:     | FileCheck -check-prefix=ARM_32 %s
; RUN: llc < %s -emulated-tls -mtriple=arm-linux-androidabi -relocation-model=pic -O3 \
; RUN:     | FileCheck -check-prefix=ARM_32 %s
; RUN: llc < %s -emulated-tls -mtriple=arm-linux-androidabi -O3 \
; RUN:     | FileCheck -check-prefix=ARM_32 %s
; RUN: llc < %s -emulated-tls -mtriple=arm-apple-darwin -O3 \
; RUN:     | FileCheck -check-prefix=DARWIN %s
; RUN: llc < %s -emulated-tls -mtriple=thumbv7-windows-gnu -O3 \
; RUN:     | FileCheck -check-prefix=WIN %s

; RUN: llc < %s -mtriple=arm-linux-android -relocation-model=pic \
; RUN:     | FileCheck -check-prefix=ARM_32 %s
; RUN: llc < %s -mtriple=arm-linux-androidabi -relocation-model=pic \
; RUN:     | FileCheck -check-prefix=ARM_32 %s
; RUN: llc < %s -mtriple=arm-linux-androidabi -relocation-model=pic -O3 \
; RUN:     | FileCheck -check-prefix=ARM_32 %s
; RUN: llc < %s -mtriple=arm-linux-androidabi -O3 \
; RUN:     | FileCheck -check-prefix=ARM_32 %s
; arm-apple-darwin must use -emulated-tls
; windows must use -emulated-tls

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
; ARM_32:        bl __emutls_get_address
; ARM_32:        .long __emutls_v.internal_y
; ARM_32-NOT:    __emutls_t.external_x
; ARM_32-NOT:    __emutls_v.external_x:
; ARM_32:        .data{{$}}
; ARM_32:        .globl __emutls_v.external_y
; ARM_32:        .p2align 2
; ARM_32-LABEL:  __emutls_v.external_y:
; ARM_32-NEXT:   .long 1
; ARM_32-NEXT:   .long 2
; ARM_32-NEXT:   .long 0
; ARM_32-NEXT:   .long __emutls_t.external_y
; ARM_32:        .section .rodata,
; ARM_32-LABEL:  __emutls_t.external_y:
; ARM_32-NEXT:   .byte 7
; ARM_32:        .data{{$}}
; ARM_32-NOT:    .globl
; ARM_32:        .p2align 2
; ARM_32-LABEL:  __emutls_v.internal_y:
; ARM_32-NEXT:   .long 8
; ARM_32-NEXT:   .long 16
; ARM_32-NEXT:   .long 0
; ARM_32-NEXT:   .long __emutls_t.internal_y
; ARM_32-LABEL:  __emutls_t.internal_y:
; ARM_32-NEXT:   .long 9
; ARM_32-NEXT:   .long 0

; WIN-LABEL:  get_external_x:
; WIN:        movw r0, :lower16:.refptr.__emutls_v.external_x
; WIN:        movt r0, :upper16:.refptr.__emutls_v.external_x
; WIN:        ldr  r0, [r0]
; WIN:        bl __emutls_get_address
; WIN-LABEL:  get_external_y:
; WIN:        movw r0, :lower16:__emutls_v.external_y
; WIN:        movt r0, :upper16:__emutls_v.external_y
; WIN:        bl __emutls_get_address
; WIN-LABEL:  get_internal_y:
; WIN:        movw r0, :lower16:__emutls_v.internal_y
; WIN:        movt r0, :upper16:__emutls_v.internal_y
; WIN:        bl __emutls_get_address
; WIN-NOT:    __emutls_t.external_x
; WIN-NOT:    __emutls_v.external_x:
; WIN:        .data{{$}}
; WIN:        .globl __emutls_v.external_y
; WIN:        .p2align 2
; WIN-LABEL:  __emutls_v.external_y:
; WIN-NEXT:   .long 1
; WIN-NEXT:   .long 2
; WIN-NEXT:   .long 0
; WIN-NEXT:   .long __emutls_t.external_y
; WIN:        .section .rdata,
; WIN-LABEL:  __emutls_t.external_y:
; WIN-NEXT:   .byte 7
; WIN:        .data{{$}}
; WIN-NOT:    .globl
; WIN:        .p2align 2
; WIN-LABEL:  __emutls_v.internal_y:
; WIN-NEXT:   .long 8
; WIN-NEXT:   .long 16
; WIN-NEXT:   .long 0
; WIN-NEXT:   .long __emutls_t.internal_y
; WIN-LABEL:  __emutls_t.internal_y:
; .quad 9 is equivalent to .long 9 .long 0
; WIN-NEXT:   .quad 9

; DARWIN-LABEL:  _get_external_x:
; DARWIN:        bl ___emutls_get_address
; DARWIN:        .long L___emutls_v.external_x$non_lazy_ptr-(LPC0_0+8)
; DARWIN-LABEL:  _get_external_y:
; DARWIN:        bl ___emutls_get_address
; DARWIN:        .long ___emutls_v.external_y-(LPC1_0+8)
; DARWIN-LABEL:  _get_internal_y:
; DARWIN:        bl ___emutls_get_address
; DARWIN:        .long ___emutls_v.internal_y-(LPC2_0+8)
; DARWIN-NOT:    ___emutls_t.external_x
; DARWIN-NOT:    ___emutls_v.external_x:
; DARWIN:        .section __DATA,__data
; DARWIN:        .globl ___emutls_v.external_y
; DARWIN:        .p2align 2
; DARWIN-LABEL:  ___emutls_v.external_y:
; DARWIN-NEXT:   .long 1
; DARWIN-NEXT:   .long 2
; DARWIN-NEXT:   .long 0
; DARWIN-NEXT:   .long ___emutls_t.external_y
; DARWIN:        .section __TEXT,__const
; DARWIN-LABEL:  ___emutls_t.external_y:
; DARWIN-NEXT:   .byte 7
; DARWIN:        .section __DATA,__data
; DARWIN-NOT:    .globl
; DARWIN:        .p2align 2
; DARWIN-LABEL:  ___emutls_v.internal_y:
; DARWIN-NEXT:   .long 8
; DARWIN-NEXT:   .long 16
; DARWIN-NEXT:   .long 0
; DARWIN-NEXT:   .long ___emutls_t.internal_y
; DARWIN-LABEL:  ___emutls_t.internal_y:
; DARWIN-NEXT:   .long 9
; DARWIN-NEXT:   .long 0
