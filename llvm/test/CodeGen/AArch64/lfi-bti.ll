; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+bti,+pauth %s -o - | FileCheck %s --check-prefixes=CHECK,LINUX
; RUN: llc -mtriple=aarch64_lfi -mattr=+bti,+pauth %s -o - | FileCheck %s --check-prefixes=CHECK,LFI

;; LFI routes all indirect branches through x28, so tail calls are not exempted
;; from BTI rules via the x16/x17 exception. Thus function entries must be BTI JC
;; and PACIASP cannot serve as an implicit BTI c landing pad.

define void @plain() "branch-target-enforcement" {
; CHECK-LABEL: plain:
; LINUX:       bti c
; LFI:         bti jc
; CHECK:       ret
  ret void
}

define void @signed() "branch-target-enforcement" "sign-return-address"="all" {
; CHECK-LABEL: signed:
; CHECK:       // %bb.0:
; LFI-NEXT:    bti jc
; CHECK-NEXT:  paciasp
  ret void
}

define void @signed_bkey() "branch-target-enforcement" "sign-return-address"="all" "sign-return-address-key"="b_key" {
; CHECK-LABEL: signed_bkey:
; CHECK:       // %bb.0:
; CHECK-NEXT:  .cfi_b_key_frame
; LFI-NEXT:    bti jc
; CHECK-NEXT:  pacibsp
  ret void
}

;; Internal functions with no indirect uses need no landing pad.
define internal void @internal() "branch-target-enforcement" {
; CHECK-LABEL: internal:
; CHECK-NOT:   bti
; CHECK:       ret
  ret void
}

define void @use_internal(ptr %p) {
  call void @internal()
  ret void
}
