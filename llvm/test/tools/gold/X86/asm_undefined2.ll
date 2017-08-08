; RegularLTO testcase
; RUN: llvm-as %s -o %t.o
; RUN: %gold -shared -m elf_x86_64 -o %t2 -plugin %llvmshlibdir/LLVMgold.so \
; RUN: %t.o --plugin-opt=save-temps -upatatino
; RUN: llvm-dis < %t2.0.5.precodegen.bc | FileCheck %s

; ThinLTO testcase
; RUN: opt -module-summary %s -o %t.o
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold.so \
; RUN:     --plugin-opt=save-temps \
; RUN:     --plugin-opt=thinlto -o %t2 %t.o
; RUN: llvm-dis < %t.o.5.precodegen.bc | FileCheck %s

; Check that foo is not internalized
; CHECK: define void @foo

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

module asm ".global patatino"
module asm ".equ patatino, foo"

declare void @patatino()

define void @foo() {
  call void @patatino()
  ret void
}
