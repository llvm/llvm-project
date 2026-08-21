; RUN: not llc < %s 2>&1 | FileCheck %s

target triple = "wasm32-unknown-unknown"

@g = addrspace(1) global i32 42 #0
; CHECK: error: definition of global 'g' cannot have import attribute

@g2 = external global i32 #0
; CHECK: error: imported global 'g2' must be in a wasm variable address space

attributes #0 = { "wasm-import-module"="js" }

