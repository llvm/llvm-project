; RUN: llc -mtriple=arm-linux-gnueabi < %s \
; RUN:     | FileCheck -check-prefix=CHECK-NONPIC -check-prefix=COMMON %s
; RUN: llc -mtriple=arm-linux-gnueabi -relocation-model=pic < %s \
; RUN:     | FileCheck -check-prefix=CHECK-PIC  -check-prefix=COMMON %s
; RUN: llc -emulated-tls -mtriple=arm-linux-gnueabi < %s \
; RUN:     | FileCheck -check-prefix=EMU -check-prefix=COMMON %s
; RUN: llc -emulated-tls -mtriple=arm-linux-gnueabi -relocation-model=pic < %s \
; RUN:     | FileCheck -check-prefix=EMU -check-prefix=COMMON %s


@external_gd = external thread_local global i32
@internal_gd = internal thread_local global i32 42

@external_ld = external thread_local(localdynamic) global i32
@internal_ld = internal thread_local(localdynamic) global i32 42

@external_ie = external thread_local(initialexec) global i32
@internal_ie = internal thread_local(initialexec) global i32 42

@external_le = external thread_local(localexec) global i32
@internal_le = internal thread_local(localexec) global i32 42

; ----- no model specified -----

define i32* @f1() {
entry:
  ret i32* @external_gd

  ; COMMON-LABEL:   f1:
  ; Non-PIC code can use initial-exec, PIC code has to use general dynamic.
  ; CHECK-NONPIC:   external_gd(GOTTPOFF)
  ; CHECK-PIC:      external_gd(TLSGD)
  ; EMU:            __emutls_get_address
}

define i32* @f2() {
entry:
  ret i32* @internal_gd

  ; COMMON-LABEL:   f2:
  ; Non-PIC code can use local exec, PIC code can use local dynamic,
  ; but that is not implemented, so falls back to general dynamic.
  ; CHECK-NONPIC:   internal_gd(TPOFF)
  ; CHECK-PIC:      internal_gd(TLSGD)
  ; EMU:            __emutls_get_address
}


; ----- localdynamic specified -----

define i32* @f3() {
entry:
  ret i32* @external_ld

  ; COMMON-LABEL:   f3:
  ; Non-PIC code can use initial exec, PIC should use local dynamic,
  ; but that is not implemented, so falls back to general dynamic.
  ; CHECK-NONPIC:   external_ld(GOTTPOFF)
  ; CHECK-PIC:      external_ld(TLSGD)
  ; EMU:            __emutls_get_address
}

define i32* @f4() {
entry:
  ret i32* @internal_ld

  ; COMMON-LABEL:   f4:
  ; Non-PIC code can use local exec, PIC code can use local dynamic,
  ; but that is not implemented, so it falls back to general dynamic.
  ; CHECK-NONPIC:   internal_ld(TPOFF)
  ; CHECK-PIC:      internal_ld(TLSGD)
  ; EMU:            __emutls_get_address
}


; ----- initialexec specified -----

define i32* @f5() {
entry:
  ret i32* @external_ie

  ; COMMON-LABEL:   f5:
  ; Non-PIC and PIC code will use initial exec as specified.
  ; CHECK-NONPIC:   external_ie(GOTTPOFF)
  ; CHECK-PIC:      external_ie(GOTTPOFF)
  ; EMU:            __emutls_get_address
}

define i32* @f6() {
entry:
  ret i32* @internal_ie

  ; COMMON-LABEL:   f6:
  ; Non-PIC code can use local exec, PIC code use initial exec as specified.
  ; CHECK-NONPIC:   internal_ie(TPOFF)
  ; CHECK-PIC:      internal_ie(GOTTPOFF)
  ; EMU:            __emutls_get_address
}


; ----- localexec specified -----

define i32* @f7() {
entry:
  ret i32* @external_le

  ; COMMON-LABEL:   f7:
  ; Non-PIC and PIC code will use local exec as specified.
  ; CHECK-NONPIC:   external_le(TPOFF)
  ; CHECK-PIC:      external_le(TPOFF)
  ; EMU:            __emutls_get_address
}

define i32* @f8() {
entry:
  ret i32* @internal_le

  ; COMMON-LABEL:   f8:
  ; Non-PIC and PIC code will use local exec as specified.
  ; CHECK-NONPIC:   internal_le(TPOFF)
  ; CHECK-PIC:      internal_le(TPOFF)
  ; EMU:            __emutls_get_address
}


; ----- emulated specified -----

; External declaration has no initializer.
; Internal definition has initializer.

; EMU-NOT:   __emutls_t.external_gd
; EMU-NOT:   __emutls_v.external_gd
; EMU:       .p2align 2
; EMU-LABEL: __emutls_v.internal_gd:
; EMU-NEXT:  .long 4
; EMU-NEXT:  .long 4
; EMU-NEXT:  .long 0
; EMU-NEXT:  .long __emutls_t.internal_gd
; EMU-LABEL: __emutls_t.internal_gd:
; EMU-NEXT:  .long 42
; EMU-NOT:   __emutls_t.external_gd

; __emutls_t and __emutls_v are the same for PIC and non-PIC modes.

; EMU-NOT:   __emutls_t.external_gd
; EMU-NOT:   __emutls_v.external_gd
; EMU:       .p2align 2
; EMU-LABEL: __emutls_v.internal_le:
; EMU-NEXT:  .long 4
; EMU-NEXT:  .long 4
; EMU-NEXT:  .long 0
; EMU-NEXT:  .long __emutls_t.internal_le
; EMU-LABEL: __emutls_t.internal_le:
; EMU-NEXT:  .long 42
; EMU-NOT:   __emutls_t.external_le
