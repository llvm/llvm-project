; RUN: llc < %s -O1 --debug-pass=Structure 2>&1 | FileCheck %s --check-prefix=O1
; RUN: llc < %s -O2 --debug-pass=Structure 2>&1 | FileCheck %s --check-prefix=O2

; Test if RegisterCoalesce pass is disabled in -O1.

; O1-NOT: Simple Register Coalescing
; O2: Simple Register Coalescing
target triple = "wasm32-unknown-unknown"

define void @test() {
  ret void
}
