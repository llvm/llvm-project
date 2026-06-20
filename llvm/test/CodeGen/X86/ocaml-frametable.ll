; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

; Instead of the default stackmap section, the OCaml backend emits a global
; "caml<module>__frametable" symbol in the data section.

; CHECK: .data
; CHECK: .globl{{.*}}__frametable
; CHECK: caml{{.*}}__frametable"
; CHECK: .quad 0
define cc128 i64 @g(i64 %x) {
  ret i64 %x
}
