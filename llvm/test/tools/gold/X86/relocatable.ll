; RUN: llvm-as %s -o %t1.o
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -r %t1.o -o %t
; RUN: llvm-readobj -symbols %t | FileCheck %s

; CHECK: Symbol {
; CHECK:   Name: foo
; CHECK-NEXT:   Value: 0x0
; CHECK-NEXT:   Size: 1
; CHECK-NEXT:   Binding: Global
; CHECK-NEXT:   Type: Function
; CHECK-NEXT:   Other: 0
; CHECK-NEXT:   Section: .text.foo
; CHECK-NEXT: }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo() {
  call void @bar()
  ret void
}

define internal void @bar() {
  ret void
}
