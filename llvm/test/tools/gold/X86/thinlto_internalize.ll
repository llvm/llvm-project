; RUN: opt -module-summary %s -o %t.o
; RUN: opt -module-summary %p/Inputs/thinlto_internalize.ll -o %t2.o

; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold.so \
; RUN:     --plugin-opt=thinlto \
; RUN:     --plugin-opt=-import-instr-limit=0 \
; RUN:     --plugin-opt=save-temps \
; RUN:     -o %t3.o %t2.o %t.o
; RUN: llvm-dis %t.o.opt.bc -o - | FileCheck %s

; f() should be internalized and eliminated after inlining
; CHECK-NOT: @f()

target triple = "x86_64-unknown-linux-gnu"
define i32 @g() {
  call void @f()
  ret i32 0
}
define void @f() {
  ret void
}
