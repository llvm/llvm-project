; RUN: llvm-as %s -o %t.o

; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t.o -o %t2.o \
; RUN:    -version-script=%p/Inputs/linker-script.export
; RUN: llvm-dis %t2.o -o - | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; CHECK: define void @f()
define void @f() {
  ret void
}

; CHECK: define internal void @g()
define void @g() {
  ret void
}
