; RUN: not llc < %s -mtriple=wasm32-unknown-unknown -filetype=obj 2>&1 | FileCheck %s

; CHECK: error: common symbols are not yet implemented for Wasm: x
; CHECK: error: common symbols are not yet implemented for Wasm: y
@x = common global i32 0, align 4
@y = common global i32 0, align 4
