; This tests that llc accepts all valid WebAssembly CPUs.

; RUN: llc < %s -asm-verbose=false -mtriple=wasm32-unknown-unknown -mcpu=mvp 2>&1 | FileCheck %s
; RUN: not llc < %s -asm-verbose=false -mtriple=wasm64-unknown-unknown-wasm -mcpu=mvp 2>&1 | FileCheck %s --check-prefix=WASM64
; RUN: llc < %s -asm-verbose=false -mtriple=wasm32-unknown-unknown -mcpu=generic 2>&1 | FileCheck %s
; RUN: not llc < %s -asm-verbose=false -mtriple=wasm64-unknown-unknown-wasm -mcpu=generic 2>&1 | FileCheck %s  --check-prefix=WASM64
; RUN: llc < %s -asm-verbose=false -mtriple=wasm32-unknown-unknown -mcpu=bleeding-edge 2>&1 | FileCheck %s
; RUN: not llc < %s -asm-verbose=false -mtriple=wasm64-unknown-unknown-wasm -mcpu=bleeding-edge 2>&1 | FileCheck %s  --check-prefix=WASM64
; RUN: llc < %s -asm-verbose=false -mtriple=wasm32-unknown-unknown -mcpu=invalidcpu 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: not llc < %s -asm-verbose=false -mtriple=wasm64-unknown-unknown-wasm -mcpu=invalidcpu 2>&1 | FileCheck %s --check-prefix=WASM64

; CHECK-NOT: is not a recognized processor for this target
; INVALID: {{.+}} is not a recognized processor for this target
; WASM64: 64-bit WebAssembly (wasm64) is not currently supported

define i32 @f(i32 %i_like_the_web) {
  ret i32 %i_like_the_web
}
