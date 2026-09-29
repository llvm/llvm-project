; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s
; Verify that types are output for external symbols, even in the absence of any defined functions

;      CHECK: .type foo,@object
; CHECK-NEXT: .section .data.foo,"",@
; CHECK-NEXT: .globl foo
@foo = local_unnamed_addr global ptr @bar, align 4

; CHECK-LABEL: foo:
;  CHECK-NEXT: .int32 bar
;  CHECK-NEXT: .size foo, 4

; CHECK: .functype bar (i32) -> (i32)
declare i32 @bar(i32 noundef)
