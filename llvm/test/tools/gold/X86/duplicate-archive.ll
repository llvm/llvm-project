; Test that the gold LTO plugin does not crash when the same archive is
; passed more than once on the command line and a member ends up claimed
; multiple times. With --whole-archive the linker pulls the member from
; every occurrence of the archive, so the plugin sees the same module
; twice. Previously this triggered an assertion failure in gold-plugin.cpp
; because the duplicate modules produced identical identifiers in
; ObjectToIndexFileState.

; RUN: llvm-as %s -o %t.o
; RUN: llvm-as %p/Inputs/duplicate-archive.ll -o %t2.o
; RUN: llvm-ar rcs %t.a %t2.o

; Full LTO: archive claimed twice via --whole-archive must not crash.
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -m elf_x86_64 -shared \
; RUN:    -o %t.so %t.o --whole-archive %t.a %t.a --no-whole-archive
; RUN: llvm-nm %t.so | FileCheck %s

; ThinLTO: same scenario must not crash.
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -m elf_x86_64 -shared \
; RUN:    --plugin-opt=thinlto \
; RUN:    -o %t.thinlto.so %t.o --whole-archive %t.a %t.a --no-whole-archive
; RUN: llvm-nm %t.thinlto.so | FileCheck %s

; CHECK-DAG: T bar
; CHECK-DAG: W foo

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @foo()

define void @bar() {
  call void @foo()
  ret void
}
