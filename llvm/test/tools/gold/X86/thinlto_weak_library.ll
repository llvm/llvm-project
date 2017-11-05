; Test to ensure that ThinLTO sorts the modules before passing to the
; final native link based on the linker's determination of which
; object within a static library contains the prevailing def of a symbol.

; First generate bitcode with a module summary index for each file
; RUN: opt -module-summary %s -o %t.o
; RUN: opt -module-summary %p/Inputs/thinlto_weak_library1.ll -o %t2.o
; RUN: opt -module-summary %p/Inputs/thinlto_weak_library2.ll -o %t3.o

; Although the objects are ordered "%t2.o %t3.o" in the library, the
; linker selects %t3.o first since it satisfies a strong reference from
; %t.o. It later selects %t2.o based on the strong ref from %t3.o.
; Therefore, %t3.o's copy of @f is prevailing, and we need to link
; %t3.o before %t2.o in the final native link.
; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=thinlto \
; RUN:    --plugin-opt=save-temps \
; RUN:    -m elf_x86_64 \
; RUN:    -o %t4 \
; RUN:    %t.o \
; RUN:    --start-lib %t2.o %t3.o --end-lib

; Make sure we completely dropped the definition of the non-prevailing
; copy of f() (and didn't simply convert to available_externally, which
; would incorrectly enable inlining).
; RUN: llvm-dis %t2.o.1.promote.bc -o - | FileCheck %s
; CHECK: declare dso_local i32 @f()

; ModuleID = 'thinlto_weak_library.c'
source_filename = "thinlto_weak_library.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define i32 @main() local_unnamed_addr {
entry:
  tail call void (...) @test2()
  ret i32 0
}

declare void @test2(...) local_unnamed_addr
