; RUN: llc %s -o %t.s
; RUN: llvm-mc %t.s -o %t.o -filetype=obj
; RUN: llvm-as %p/Inputs/drop-linkage.ll -o %t2.o

; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t.o %t2.o -o %t3.o
; RUN: llvm-dis %t3.o -o - | FileCheck %s

define void @foo() {
  ret void
}

; CHECK: declare extern_weak void @foo(){{$}}
