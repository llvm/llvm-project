; RUN: llvm-as %s -o %t.bc
; RUN: llvm-as %p/Inputs/comdat2.ll -o %t2.bc
; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t.bc %t2.bc -o %t3.bc
; RUN: llvm-dis %t3.bc -o - | FileCheck %s


$foo = comdat any
@foo = global i8 0, comdat

; CHECK: @foo = global i8 0, comdat

; CHECK: define void @zed() {
; CHECK:   call void @bar()
; CHECK:   ret void
; CHECK: }

; CHECK: declare void @bar()
