; RUN: llc -mattr=harden-sls-blr,harden-sls-nocomdat -verify-machineinstrs -mtriple=aarch64-apple-macos < %s | FileCheck %s
; RUN: llc -mattr=harden-sls-blr,harden-sls-nocomdat -mtriple=aarch64-apple-macos -filetype=obj < %s \
; RUN:   | llvm-nm - | FileCheck %s --check-prefix=OBJ --implicit-check-not=slsblr

; On Mach-O, every symbol carries a leading '_', and the BL to an SLS BLR thunk
; has to name the thunk with it too, or the call is left undefined.

define void @indirect_call(ptr %f) {
; CHECK-LABEL: _indirect_call:
; CHECK:         bl ___llvm_slsblr_thunk_x0
entry:
  call void %f()
  ret void
}

; CHECK-LABEL: ___llvm_slsblr_thunk_x0:
; CHECK-NEXT:  ; %bb.0:
; CHECK-NEXT:    mov x16, x0
; CHECK-NEXT:    br x16
; CHECK-NEXT:    dsb sy
; CHECK-NEXT:    isb

; OBJ: t ___llvm_slsblr_thunk_x0
