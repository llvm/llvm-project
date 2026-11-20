; RUN: split-file %s %t
; RUN: not llc < %t/vec.ll -mtriple=wasm32-unknown-unknown -mattr=+simd128 -filetype=obj -o /dev/null 2>&1 | FileCheck %s --check-prefix=V128
; RUN: not llc < %t/fref.ll -mtriple=wasm32-unknown-unknown -mattr=+reference-types -filetype=obj -o /dev/null 2>&1 | FileCheck %s --check-prefix=FUNCREF

; V128: error: WebAssembly global 'vec' has a type that cannot be initialized
; FUNCREF: error: WebAssembly global 'fref' has a type that cannot be initialized

;--- vec.ll
@vec = addrspace(1) global <4 x i32> zeroinitializer

;--- fref.ll
@fref = addrspace(1) global target("wasm.funcref") zeroinitializer
