; RUN: llc -mtriple=aarch64-linux-gnu %s -o - | FileCheck %s --check-prefixes=CHECK,LINUX
; RUN: llc -mtriple=aarch64_lfi %s -o - | FileCheck %s --check-prefixes=CHECK,LFI

;; LFI routes all indirect branches through x28, so tail calls are not exempted
;; from BTI rules via the x16/x17 exception. Thus function entries be BTI JC
;; and PACIASP cannot serve as an implicit BTI c landing pad.

define void @plain() "branch-target-enforcement" {
; CHECK-LABEL: plain:
; LINUX:       hint #34
; LFI:         hint #38
; CHECK:       ret
  ret void
}

define void @signed() "branch-target-enforcement" "sign-return-address"="all" {
; CHECK-LABEL: signed:
; CHECK:       // %bb.0:
; LFI-NEXT:    hint #38
; CHECK-NEXT:  hint #25
  ret void
}

define void @signed_bkey() "branch-target-enforcement" "sign-return-address"="all" "sign-return-address-key"="b_key" {
; CHECK-LABEL: signed_bkey:
; CHECK:       // %bb.0:
; CHECK-NEXT:  .cfi_b_key_frame
; LFI-NEXT:    hint #38
; CHECK-NEXT:  hint #27
  ret void
}

;; Internal functions with no indirect uses need no landing pad.
define internal void @internal() "branch-target-enforcement" {
; CHECK-LABEL: internal:
; CHECK-NOT:   hint #3{{[468]}}
; CHECK:       ret
  ret void
}

define void @use_internal(ptr %p) {
  call void @internal()
  ret void
}
