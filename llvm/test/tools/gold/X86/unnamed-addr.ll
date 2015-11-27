; RUN: llvm-as %s -o %t.o

; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t.o -o %t2.o
; RUN: llvm-dis %t2.o -o - | FileCheck %s

@a = internal unnamed_addr constant i8 42

define i8* @f() {
  ret i8* @a
}

; CHECK: @a = internal unnamed_addr constant i8 42
