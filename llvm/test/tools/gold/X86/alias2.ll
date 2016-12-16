; RUN: llvm-as %s -o %t.o
; RUN: %gold -shared -o %t2.bc -plugin %llvmshlibdir/LLVMgold.so %t.o -plugin-opt=emit-llvm
; RUN: llvm-dis %t2.bc -o - | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@bar = alias void (), void ()* @zed
define void @foo() {
  call void @bar()
  ret void
}
define void @zed() {
  ret void
}

; CHECK: @bar = alias void (), void ()* @zed

; CHECK:      define void @foo() {
; CHECK-NEXT:   call void @bar()
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK:      define void @zed() {
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
