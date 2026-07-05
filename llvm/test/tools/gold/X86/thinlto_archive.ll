; Generate summary sections
; RUN: opt -module-summary %s -o %t.o
; RUN: opt -module-summary %p/Inputs/thinlto_archive1.ll -o %t2.o
; RUN: opt -module-summary %p/Inputs/thinlto_archive2.ll -o %t3.o

; Generate the static library
; RUN: llvm-ar --format=gnu r %t.a %t2.o %t3.o

; Test importing from archive library via gold, using jobs=1 to ensure
; output messages are not interleaved.
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -m elf_x86_64 \
; RUN:    --plugin-opt=thinlto \
; RUN:    --plugin-opt=-print-imports \
; RUN:    --plugin-opt=jobs=1 \
; RUN:    -shared %t.o %t.a -o %t4 2>&1 | FileCheck %s
; RUN: llvm-nm %t4 | FileCheck %s --check-prefix=NM

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-DAG: Import g
declare void @g(...)
; CHECK-DAG: Import h
declare void @h(...)

; NM: T f
define void @f() {
entry:
  call void (...) @g()
  call void (...) @h()
  ret void
}
