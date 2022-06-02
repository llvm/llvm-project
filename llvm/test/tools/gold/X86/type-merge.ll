; RUN: llvm-as %s -o %t.o
; RUN: llvm-as %p/Inputs/type-merge.ll -o %t2.o
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t.o %t2.o -o %t3.o
; RUN: llvm-dis %t3.o -o - | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo()  {
  call void @bar(ptr null)
  ret void
}
declare void @bar(ptr)

; CHECK:      define void @foo() {
; CHECK-NEXT:   call void @bar(ptr null)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: declare void @bar(ptr)

; CHECK:      define void @zed() {
; CHECK-NEXT:   call void @bar()
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
