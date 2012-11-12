; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck -check-prefix=CHECK-NO-FP %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin -disable-fp-elim | FileCheck -check-prefix=CHECK-FP %s

define void @func() {
entry:
  unreachable
}
; CHECK-NO-FP:     _func:
; CHECK-NO-FP-NEXT: .cfi_startproc
; CHECK-NO-FP:     nop
; CHECK-NO-FP-NEXT: .cfi_endproc

; CHECK-FP:      _func:
; CHECK-FP-NEXT: .cfi_startproc
; CHECK-FP-NEXT: :
; CHECK-FP-NEXT: pushq %rbp
; CHECK-FP-NEXT: :
; CHECK-FP-NEXT: .cfi_def_cfa_offset 16
; CHECK-FP-NEXT: :
; CHECK-FP-NEXT: .cfi_offset %rbp, -16
; CHECK-FP-NEXT: movq %rsp, %rbp
; CHECK-FP-NEXT: :
; CHECK-FP-NEXT: .cfi_def_cfa_register %rbp
; CHECK-FP-NEXT: nop
; CHECK-FP-NEXT: .cfi_endproc
