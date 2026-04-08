; Test that passing the same archive twice to the LTO gold plugin does not
; crash. This is a common pattern used to resolve circular dependencies
; between archives (e.g., "ld -la -lb -la"). Previously this triggered an
; assertion failure in gold-plugin.cpp because duplicate archive members
; produced identical module identifiers in ObjectToIndexFileState.

; RUN: llvm-as %s -o %t.o
; RUN: llvm-as %p/Inputs/duplicate-archive.ll -o %t2.o
; RUN: llvm-ar rcs %t.a %t2.o

; Full LTO: same archive passed twice should not crash.
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -m elf_x86_64 -shared \
; RUN:    -o %t.so %t.o %t.a %t.a
; RUN: llvm-nm %t.so | FileCheck %s

; ThinLTO: same archive passed twice should also not crash.
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -m elf_x86_64 -shared \
; RUN:    --plugin-opt=thinlto \
; RUN:    -o %t.thinlto.so %t.o %t.a %t.a
; RUN: llvm-nm %t.thinlto.so | FileCheck %s

; CHECK-DAG: T foo
; CHECK-DAG: T bar

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @foo()

define void @bar() {
  call void @foo()
  ret void
}
