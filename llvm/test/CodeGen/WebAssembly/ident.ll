; RUN: llc < %s -asm-verbose=false -wasm-keep-registers | FileCheck %s

; Test llvm.ident.

target triple = "wasm32-unknown-unknown"

; CHECK: .ident "hello world"

!llvm.ident = !{!0}

!0 = !{!"hello world"}
