; RUN: opt -module-hash -module-summary %s -o %t.bc
; RUN: opt -module-hash -module-summary %p/Inputs/empty_module_with_cache.ll -o %t2.bc

; Verify that enabling caching is working, even if the module is empty
; RUN: rm -Rf %t.cache && mkdir %t.cache
; RUN: llvm-lto -thinlto-action=run %t2.bc  %t.bc -exported-symbol=main -thinlto-cache-dir %t.cache
; RUN: ls %t.cache/llvmcache.timestamp
; RUN: ls %t.cache | count 3

; Verify that enabling caching is working with llvm-lto2
; RUN: rm -Rf %t.cache
; RUN: llvm-lto2 run -o %t.o %t2.bc  %t.bc -cache-dir %t.cache \
; RUN:  -r=%t2.bc,_main,plx
; RUN: ls %t.cache | count 2

; Same, but without hash, the index will be empty and caching should not happen

; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/empty_module_with_cache.ll -o %t2.bc

; Verify that caching is disabled for module without hash
; RUN: rm -Rf %t.cache && mkdir %t.cache
; RUN: llvm-lto -thinlto-action=run %t2.bc  %t.bc -exported-symbol=main -thinlto-cache-dir %t.cache
; RUN: ls %t.cache/llvmcache.timestamp
; RUN: ls %t.cache | count 1

; Verify that caching is disabled for module without hash, with llvm-lto2
; RUN: rm -Rf %t.cache
; RUN: llvm-lto2 run -o %t.o %t2.bc  %t.bc -cache-dir %t.cache \
; RUN:  -r=%t2.bc,_main,plx
; RUN: ls %t.cache | count 0


target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"
