; RUN: not llc -mtriple=wasm32-unknown-unknown -filetype=asm %s -o - 2>&1 | FileCheck %s

; CHECK: common symbols are not yet implemented for Wasm: x
; CHECK: common symbols are not yet implemented for Wasm: y
@x = common global i32 0, align 4
@y = common global i32 0, align 4
