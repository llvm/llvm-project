; RUN: llc -mtriple=wasm32-unknown-unknown -filetype=asm %s -o - | FileCheck %s

; CHECK:         .type   x,@object
; CHECK-NEXT:    .comm   x,4,2
; CHECK:         .type   y,@object
; CHECK-NEXT:    .comm   y,8,3

@x = common global i32 0, align 4
@y = common global i64 0, align 8
