; RUN: llvm-as %s -o %t.o
; RUN: llvm-as %p/Inputs/visibility.ll -o %t2.o

; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    -m elf_x86_64 \
; RUN:    --plugin-opt=save-temps \
; RUN:    -shared %t.o %t2.o -o %t.so
; RUN: llvm-readobj -t %t.so | FileCheck %s
; RUN: llvm-dis %t.so.0.2.internalize.bc -o - | FileCheck --check-prefix=IR %s

; CHECK:      Name: foo
; CHECK-NEXT: Value:
; CHECK-NEXT: Size:
; CHECK-NEXT: Binding: Global
; CHECK-NEXT: Type: Function
; CHECK-NEXT: Other [
; CHECK-NEXT:   STV_PROTECTED
; CHECK-NEXT: ]

; IR: define void @foo

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define weak protected void @foo() {
  ret void
}
