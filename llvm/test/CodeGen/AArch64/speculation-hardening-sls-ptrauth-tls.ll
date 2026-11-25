; RUN: llc -verify-machineinstrs -mtriple=arm64e-apple-darwin -mattr=+harden-sls-nocomdat -O2 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=arm64e-apple-darwin -mattr=+harden-sls-nocomdat -global-isel -O2 < %s | FileCheck %s

; Verify that Darwin ptrauth TLS calls lower through an SLS thunk without
; triggering the BLR-only assertion in AArch64InstructionSelector.

@tls_var = thread_local global i32 0

define ptr @get_tls() #0 {
; CHECK-LABEL: _get_tls:
; CHECK:       bl __llvm_slsblr_thunk_aaz_x[[REG:[0-9]+]]
; CHECK-NOT:   blraaz
; CHECK:       ___llvm_slsblr_thunk_aaz_x[[REG]]:
; CHECK:       mov x16, x[[REG]]
; CHECK-NEXT:  braaz x16
; CHECK-NEXT:  dsb sy
; CHECK-NEXT:  isb
  ret ptr @tls_var
}

attributes #0 = { nounwind "ptrauth-calls" "target-features"="+pauth,+harden-sls-blr" }
