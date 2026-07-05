; RUN: llvm-as %p/Inputs/emit-llvm.foo.ll -o %t.foo.bc
; RUN: llvm-as %p/Inputs/emit-llvm.bar.ll -o %t.bar.bc
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext --shared -plugin-opt thinlto -plugin-opt emit-llvm -m elf_x86_64 %t.foo.bc %t.bar.bc -o %t.bc
; RUN: llvm-dis %t.bc1 -o - | FileCheck --check-prefix=CHECK-BC1 %s
; RUN: llvm-dis %t.bc2 -o - | FileCheck --check-prefix=CHECK-BC2 %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-BC1: define dso_local i32 @_Z3foov()
define dso_local i32 @_Z3foov() {
    ret i32 0
}
; CHECK-BC2: define dso_local i32 @_Z3barv()
define dso_local i32 @_Z3barv() {
    ret i32 0
}
