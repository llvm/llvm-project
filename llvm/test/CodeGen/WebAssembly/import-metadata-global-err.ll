; RUN: not llc < %s 2>&1 | FileCheck %s

target triple = "wasm32-unknown-unknown"

@g = global i32 42, !wasm.import.module !0

; CHECK: error: definition of global 'g' cannot have import metadata

!0 = !{!"js"}
