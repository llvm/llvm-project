; Because some arguments are passed by reference (through stack),
; the compiler should not do tail-call optimization.
; RUN: llc -mtriple=aarch64 -mattr=+sve < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

; CHECK-LABEL: caller:
; CHECK:       addvl sp, sp, #-[[STACKSIZE:[0-9]+]]
; CHECK-NOT:   addvl sp
; CHECK:       bl callee
; CHECK:       addvl sp, sp, #[[STACKSIZE]]
; CHECK:       ret
define <vscale x 16 x i8> @caller(<vscale x 16 x i8> %v) {
  %1 = tail call <vscale x 16 x i8> @callee(<vscale x 16 x i8> %v, <vscale x 16 x i8> %v, <vscale x 16 x i8> %v, <vscale x 16 x i8> %v, <vscale x 16 x i8> %v, <vscale x 16 x i8> %v, <vscale x 16 x i8> %v, <vscale x 16 x i8> %v, <vscale x 16 x i8> %v)
  ret <vscale x 16 x i8> %1
}

declare <vscale x 16 x i8> @callee(<vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>)

; CHECK-LABEL: caller_pred:
; CHECK:       addvl sp, sp, #-[[STACKSIZE:[0-9]+]]
; CHECK-NOT:   addvl sp
; CHECK:       bl callee_pred
; CHECK:       addvl sp, sp, #[[STACKSIZE]]
; CHECK:       ret
define <vscale x 16 x i1> @caller_pred(<vscale x 16 x i1> %v) {
  %1 = tail call <vscale x 16 x i1> @callee_pred(<vscale x 16 x i1> %v, <vscale x 16 x i1> %v, <vscale x 16 x i1> %v, <vscale x 16 x i1> %v, <vscale x 16 x i1> %v)
  ret <vscale x 16 x i1> %1
}

declare <vscale x 16 x i1> @callee_pred(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>)
