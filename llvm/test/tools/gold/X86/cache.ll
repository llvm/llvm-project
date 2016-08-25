; RUN: opt -module-summary %s -o %t.o
; RUN: opt -module-summary %p/Inputs/cache.ll -o %t2.o

; Verify that enabling caching is working
; RUN: rm -Rf %t.cache && mkdir %t.cache
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold.so \
; RUN:     --plugin-opt=thinlto \
; RUN:     --plugin-opt=cache-dir=%t.cache \
; RUN:     -o %t3.o %t2.o %t.o

; RUN: ls %t.cache | count 2

target triple = "x86_64-unknown-linux-gnu"

define void @globalfunc() #0 {
entry:
  ret void
}
