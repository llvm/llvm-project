; RUN: llc -mtriple=aarch64-linux-gnu %s -o - | FileCheck %s --check-prefixes=CHECK,NOBTI
; RUN: llc -mtriple=aarch64-none-elf %s -o - | FileCheck %s --check-prefixes=CHECK,NOBTI
; RUN: llc -mtriple=aarch64-none-macho %s -o - | FileCheck %s --check-prefixes=CHECK,BTI
; RUN: llc -mtriple=aarch64-windows-msvc %s -o - | FileCheck %s --check-prefixes=CHECK,BTI

;; This function has internal linkage, and nothing in this translation unit
;; calls it indirectly. So it doesn't need a BTI at the start ... except that
;; it might, if at link time if the linker inserts a long-branch thunk using a
;; BLR instruction.
;;
;; For ELF targets, both Linux and bare-metal, we expect no BTI instruction at
;; the start of the function, because AAELF64 specifies that it's not needed:
;; if the linker wants to do that then it's responsible for making a 'landing
;; pad' near the target function which _does_ have a BTI, and pointing the
;; indirect call at that.
;;
;; But this is specified in AAELF64, so non-ELF targets can't rely on that
;; guarantee, and we expect LLVM to insert the BTI anyway.
define internal void @internal_linkage() "branch-target-enforcement" {
; CHECK-LABEL: internal_linkage:
; BTI:         hint #34
; NOBTI-NOT:   hint #34
; CHECK:       ret
entry:
  ret void
}

;; This function has internal linkage but _is_ potentially called indirectly
;; (its address escapes from the module via external_linkage() below), so it
;; needs a BTI irrespective of target triple.
define internal void @indirectly_called() "branch-target-enforcement" {
; CHECK-LABEL: indirectly_called:
; CHECK:       hint #34
; CHECK:       ret
entry:
  ret void
}

;; This function has external linkage, so it needs a BTI in all circumstances.
define ptr @external_linkage() "branch-target-enforcement" {
; CHECK-LABEL: external_linkage:
; CHECK:       hint #34
; CHECK:       ret
entry:
  call void @internal_linkage()
  ret ptr @indirectly_called
}
