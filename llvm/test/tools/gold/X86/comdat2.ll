; RUN: llvm-as %s -o %t.bc
; RUN: llvm-as %p/Inputs/comdat2.ll -o %t2.bc
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t.bc %t2.bc -o %t3.bc
; RUN: llvm-dis %t3.bc -o - | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

$foo = comdat any
@foo = global i8 0, comdat

; CHECK: @foo = global i8 0, comdat

; CHECK: define void @zed() {
; CHECK:   call void @bar()
; CHECK:   ret void
; CHECK: }

; CHECK: declare void @bar()
