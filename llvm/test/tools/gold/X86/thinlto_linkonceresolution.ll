; RUN: opt -module-summary %s -o %t.o
; RUN: opt -module-summary %p/Inputs/thinlto_linkonceresolution.ll -o %t2.o

; Ensure the plugin ensures that for ThinLTO the prevailing copy of a
; linkonce symbol is changed to weak to ensure it is not eliminated.
; Note that gold picks the first copy of f() as the prevailing one,
; so listing %t2.o first is sufficient to ensure that this copy is
; preempted. Also, set the import-instr-limit to 0 to prevent f() from
; being imported from %t2.o which hides the problem.
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:     --plugin-opt=thinlto \
; RUN:     --plugin-opt=-import-instr-limit=0 \
; RUN:     --plugin-opt=save-temps \
; RUN:     -shared \
; RUN:     -o %t3.o %t2.o %t.o
; RUN: llvm-nm %t3.o | FileCheck %s
; RUN: llvm-dis %t.o.4.opt.bc -o - | FileCheck --check-prefix=OPT %s
; RUN: llvm-dis %t2.o.4.opt.bc -o - | FileCheck --check-prefix=OPT2 %s

; Ensure that f() is defined in resulting object file, and also
; confirm the weak linkage directly in the saved opt bitcode files.
; CHECK-NOT: U f
; OPT-NOT: @f()
; OPT2: define weak_odr hidden void @f()

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
define i32 @g() {
  call void @f()
  ret i32 0
}
define linkonce_odr hidden void @f() {
  ret void
}
