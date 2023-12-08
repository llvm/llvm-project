; RUN: llc < %s -mtriple=x86_64-linux-gnu | FileCheck %s

; CHECK:        .text
; CHECK-NEXT:   .file   "<stdin>"

define i32 @main(i32 %x) nounwind gc "ocaml" {
; CHECK:   .globl "caml<stdin>__code_begin"
; CHECK-NEXT: "caml<stdin>__code_begin":
; CHECK-NEXT:   .data
; CHECK-NEXT:   .globl  "caml<stdin>__data_begin"
; CHECK-NEXT: "caml<stdin>__data_begin":

  %puts = tail call i32 @foo(i32 %x)
  ret i32 0

; CHECK:        .globl "caml<stdin>__code_end"
; CHECK-NEXT: "caml<stdin>__code_end":
; CHECK-NEXT:   .data
; CHECK-NEXT:   .globl "caml<stdin>__data_end"
; CHECK-NEXT: "caml<stdin>__data_end":
; CHECK-NEXT:   .quad   0
; CHECK-NEXT:   .globl "caml<stdin>__frametable"
; CHECK-NEXT: "caml<stdin>__frametable":
; CHECK-NEXT:   .short  1
; CHECK-NEXT:   .p2align  3
; CHECK-NEXT:                # live roots for main
; CHECK-NEXT:   .quad   .Ltmp0
; CHECK-NEXT:   .short  8
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .p2align  3
}

declare i32 @foo(i32)
