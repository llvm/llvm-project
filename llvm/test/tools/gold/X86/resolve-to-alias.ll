; RUN: llvm-as %s -o %t.o
; RUN: llvm-as %p/Inputs/resolve-to-alias.ll -o %t2.o

; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t.o %t2.o -o %t.bc
; RUN: llvm-dis %t.bc -o %t.ll
; RUN: FileCheck --check-prefix=PASS1 %s < %t.ll
; RUN: FileCheck --check-prefix=PASS2 %s < %t.ll

; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t2.o %t.o -o %t.bc
; RUN: llvm-dis %t.bc -o %t.ll
; RUN: FileCheck --check-prefix=PASS1 %s < %t.ll
; RUN: FileCheck --check-prefix=PASS2 %s < %t.ll

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo() {
  call void @bar()
  ret void
}
declare void @bar()

; PASS1: @bar = alias void (), ptr @zed

; PASS1:      define void @foo() {
; PASS1-NEXT:   call void @bar()
; PASS1-NEXT:   ret void
; PASS1-NEXT: }

; PASS2:      define void @zed() {
; PASS2-NEXT:   ret void
; PASS2-NEXT: }
