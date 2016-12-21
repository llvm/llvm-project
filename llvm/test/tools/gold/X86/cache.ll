; Verify that enabling caching is ignoring module when we emit them without hash.
; RUN: opt -module-summary %s -o %t.o
; RUN: opt -module-summary %p/Inputs/cache.ll -o %t2.o

; RUN: rm -Rf %t.cache && mkdir %t.cache
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold.so \
; RUN:     --plugin-opt=thinlto \
; RUN:     --plugin-opt=cache-dir=%t.cache \
; RUN:     -o %t3.o %t2.o %t.o

; RUN: ls %t.cache | count 0


; Verify that enabling caching is working with module with hash.

; RUN: opt -module-hash -module-summary %s -o %t.o
; RUN: opt -module-hash -module-summary %p/Inputs/cache.ll -o %t2.o

; RUN: rm -Rf %t.cache && mkdir %t.cache
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold.so \
; RUN:     --plugin-opt=thinlto \
; RUN:     --plugin-opt=cache-dir=%t.cache \
; RUN:     -o %t3.o %t2.o %t.o

; RUN: ls %t.cache | count 2

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @globalfunc() #0 {
entry:
  ret void
}
