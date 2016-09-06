; RUN: llc < %s -mtriple=arm64-eabi | FileCheck %s

; ARM64ISelLowering.cpp was creating a new (floating-point) load for efficiency
; but not updating chain-successors of the old one. As a result, the two memory
; operations in this function both ended up direct successors to the EntryToken
; and could be reordered.

@var = global i32 0, align 4

define float @foo() {
; CHECK-LABEL: foo:
  ; Load must come before we clobber @var
; CHECK: adrp x[[VARBASE:[0-9]+]], {{_?var}}
; CHECK: ldr [[SREG:s[0-9]+]], [x[[VARBASE]],
; CHECK: str wzr, [x[[VARBASE]],

  %val = load i32, i32* @var, align 4
  store i32 0, i32* @var, align 4

  %fltval = sitofp i32 %val to float
  ret float %fltval
}
