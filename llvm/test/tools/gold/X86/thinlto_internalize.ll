; RUN: opt -module-summary %s -o %t.o
; RUN: opt -module-summary %p/Inputs/thinlto_internalize.ll -o %t2.o

; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:     --plugin-opt=thinlto \
; RUN:     --plugin-opt=-import-instr-limit=0 \
; RUN:     --plugin-opt=save-temps \
; RUN:     -o %t3.o %t2.o %t.o
; RUN: llvm-dis %t.o.4.opt.bc -o - | FileCheck %s

; f() should be internalized and eliminated after inlining
; CHECK-NOT: @f()

; h() should be internalized after promotion, and eliminated after inlining
; CHECK-NOT: @h.llvm.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
define i32 @g() {
  call void @f()
  call void @h()
  ret i32 0
}
define void @f() {
  ret void
}
define internal void @h() {
  ret void
}
