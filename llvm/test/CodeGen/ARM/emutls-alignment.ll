; RUN: llc -emulated-tls -mtriple=armv7-linux-android -relocation-model=pic < %s | FileCheck %s

; Test that emulated TLS uses preferred alignment for variables.
; This fixes a bug where emutls would use lower alignment than expected
; by the code generator, causing crashes with vectorized accesses.
; Fixes https://github.com/llvm/llvm-project/issues/167219

; A 64-byte array should get 16-byte alignment (preferred for ARM NEON).
@large_array = internal thread_local global [64 x i8] c"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"

define ptr @get_large_array() {
entry:
  ret ptr @large_array
}

; CHECK-LABEL: __emutls_v.large_array:
; CHECK-NEXT:   .long 64
; CHECK-NEXT:   .long 16
; CHECK-NEXT:   .long 0
; CHECK-NEXT:   .long __emutls_t.large_array
