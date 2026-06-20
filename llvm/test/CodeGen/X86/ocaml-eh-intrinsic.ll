; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

; The OCaml exception-handling intrinsic always lowers to the constant 0.

; CHECK-LABEL: t:
; CHECK: xorl %eax, %eax
; CHECK: retq
define i32 @t() {
  %x = call i32 @llvm.eh.ocaml.try()
  ret i32 %x
}

declare i32 @llvm.eh.ocaml.try()
